# COPY: SAN/san/model/clip_utils/visual.py
import typing as t
import warnings
from dataclasses import dataclass, field

import torch
from open_clip import CLIP
# from detectron2.layers import ShapeSpec
from open_clip.transformer import VisionTransformer, ResidualAttentionBlock
from torch import nn
from torch.nn import functional as F  # noqa
from torch.types import _dtype as DType

from .attn_helper import cross_attn_layer, downsample2d, resize_pos_embed2d
from ..sam import SamAnns

__all__ = ["FeatureExtractor", "RecWithAttnbiasHead", "SemWithMaskPooling", "NoAttCLIPLayers", "ClipOutput",
           "SCLIPLayers", "DALayers"]


def _freeze(model: nn.Module, frozen_exclude: t.Sequence[str] | None):
    if frozen_exclude is None:
        frozen_exclude = []

    if "all" in frozen_exclude:
        return
    for name, param in model.named_parameters():
        if not any([exclude in name for exclude in frozen_exclude]):
            param.requires_grad = False


def _canonical_mask(
        mask: t.Optional[torch.Tensor],
        mask_name: str,
        other_type: t.Optional[DType],
        other_name: str,
        target_type: DType,
        check_other: bool = True,
) -> t.Optional[torch.Tensor]:
    """Modified from torch.nn.functional._canonical_mask"""
    if mask is not None:
        _mask_dtype = mask.dtype
        _mask_is_float = torch.is_floating_point(mask)
        if _mask_dtype != torch.bool and not _mask_is_float:
            raise AssertionError(
                f"only bool and floating types of {mask_name} are supported")
        if check_other and other_type is not None:
            if _mask_dtype != other_type:
                warnings.warn(
                    f"Support for mismatched {mask_name} and {other_name} "
                    "is deprecated. Use same type for both instead."
                )
        if not _mask_is_float:
            mask = (
                torch.zeros_like(mask, dtype=target_type)
                .masked_fill_(mask, float("-inf"))
            )
        else:  # CHANGE: 将mask的float类型转为target_type。
            mask = mask.to(dtype=target_type)
    return mask


@dataclass
class ClipOutput(object):
    spacial_shape: tuple[int, int] = None
    idxes: list[int] = field(default_factory=list)
    clip_feat: dict[int, torch.Tensor] = field(default_factory=dict)
    img_feat: dict[int, torch.Tensor] = field(default_factory=dict)
    cls_token: dict[int, torch.Tensor] = field(default_factory=dict)
    imgs: torch.Tensor = None
    masks: list[torch.Tensor] = None
    backbone_att_biases: list[torch.Tensor] = None
    head_att_biases: list[torch.Tensor] = None
    patch_emb: torch.Tensor = None
    mask_embs: list[torch.Tensor] | torch.Tensor = None
    _low_res_method_masks: dict[str, list[torch.Tensor]] = field(default_factory=dict, init=False, repr=False)
    aff_bar_out: 'DALayers.AffBarOut' = None
    sps: list[SamAnns] = None
    leveled_patch_embs: tuple[torch.Tensor] = None

    def save(self, idx: int | None, clip_feat: torch.Tensor):
        assert self.spacial_shape is not None, "spacial_shape should be set before saving."

        if idx is None:
            idx = self.last_idx + 1

        self.idxes.append(idx)

        l, n, c = clip_feat.shape

        self.clip_feat[idx] = clip_feat
        self.img_feat[idx] = clip_feat[1:].permute(1, 2, 0).reshape(n, c, *self.spacial_shape)  # n, c, h, w
        self.cls_token[idx] = clip_feat[0:1]  # 1, n, c

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.clip_feat[idx]

    @property
    def last_idx(self) -> int:
        return max(self.idxes) if self.idxes else -1

    def get_low_res_mask(self, downsample_method: str='nearest') -> list[torch.Tensor]:
        if downsample_method in self._low_res_method_masks:
            return self._low_res_method_masks[downsample_method]

        low_res_masks = []
        for m in self.masks:
            low_res_masks.append(downsample2d(
                m.unsqueeze(0).to(dtype=torch.float32),  # [1,M,H,W]
                self.spacial_shape,  # [h,w]
                method=downsample_method
            )[0])  # [M,h,w])

        self._low_res_method_masks[downsample_method] = low_res_masks

        return low_res_masks


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        clip: CLIP,
        last_layer_idx: int | None = -1,
        frozen_exclude: t.Sequence[str]=None,
    ):
        super().__init__()
        visual_encoder = clip.visual
        self.output_tokens = visual_encoder.output_tokens  # 无用
        self.image_size = visual_encoder.image_size  # 无用
        self.patch_size = visual_encoder.patch_size
        self.grid_size = visual_encoder.grid_size
        self.num_features = visual_encoder.ln_pre.normalized_shape[0]

        self.conv1 = visual_encoder.conv1

        # class embeddings and positional embeddings
        self.class_embedding = visual_encoder.class_embedding
        self.positional_embedding = visual_encoder.positional_embedding
        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = visual_encoder.patch_dropout
        self.ln_pre = visual_encoder.ln_pre
        # if last_layer_idx == -1:  # 修改SAN，使其支持负数。如果要所有层，可以[:None].
        #     self.resblocks: list[ResidualAttentionBlock]visual_encoder.transformer.resblocks
        #     self.last_output_idx = len(self.resblocks) + 1
        # else:
        self.resblocks: list[ResidualAttentionBlock] = visual_encoder.transformer.resblocks[:last_layer_idx]  # 对ViT-B，12层resblocks取到0-8层。
        self.last_output_idx = last_layer_idx + 1
        #
        self.frozen_exclude = frozen_exclude if frozen_exclude is not None else ()
        self._freeze(self.frozen_exclude)

    def forward(self, x: torch.Tensor | ClipOutput, att_biases: list[torch.Tensor]=None) -> ClipOutput:
        if isinstance(x, ClipOutput):
            outputs = x

            assert att_biases is None, "att_biases should be None when x is ClipOutput."
            att_biases = x.backbone_att_biases
            x = x.imgs
        else:
            outputs = None

        if att_biases is None:
            att_biases = [None] * len(self.resblocks)
        else:
            assert len(att_biases) == len(self.resblocks), f"{len(att_biases)=}≠{len(self.resblocks)=}"

        x = self.conv1(x)  # shape = [*, width, grid, grid]
        _, _, h, w = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device  # 用于将cls token形状改为(n,1,c)。
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        pos_embed = self.positional_embedding.to(x.dtype)
        pos_embed = resize_pos_embed2d(pos_embed[None, ...], self.grid_size, (h, w))[0]
        x = x + pos_embed  # 特征与插值后的位置编码相加。

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was
        # passed in
        x = self.patch_dropout(x)  # 无patch_dropout。
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND

        outputs = ClipOutput(spacial_shape=(h, w)) if outputs is None else outputs
        outputs.save(0, x)
        for i, (resblock, att_bias) in enumerate(zip(self.resblocks, att_biases, strict=True), start=1):
            assert att_bias is None or torch.is_floating_point(att_bias), f"{att_bias.dtype=}"
            x = resblock(x, attn_mask=att_bias)
            outputs.save(i, x)  # 记录下CLIP每一层的image和cls token。
        return outputs

    def _freeze(self, frozen_exclude: t.Sequence[str]):
        _freeze(self, frozen_exclude)

    @property
    def size_divisibility(self):
        return self.patch_size[0]


class RecWithAttnbiasHead(nn.Module):

    def __init__(
        self,
        clip: CLIP,
        first_layer_idx: int = 0,
        frozen_exclude: list[str] | str = None,
        sos_token_format: str = "cls_token",  # 默认用cls token作为sos token。
        sos_token_num: int = None,  # 每个sos token对应一个query/mask。
        cross_attn: bool = False,
        downsample_method: str = "max",
    ):
        super().__init__()
        visual_encoder: VisionTransformer = clip.visual
        self.output_tokens = visual_encoder.output_tokens  # 一般False
        self.output_dim = visual_encoder.output_dim  # 512
        self.first_layer_idx = first_layer_idx  # SAN中为9。
        self.cross_attn = cross_attn
        self.downsample_method = downsample_method  # SAN中为max。

        self.resblocks: list[ResidualAttentionBlock] = visual_encoder.transformer.resblocks[first_layer_idx:]  # CLIP剩余的resblocks，对Vit-B，是9-11层。
        self.ln_post = visual_encoder.ln_post
        self.proj = visual_encoder.proj

        self.sos_token_format = sos_token_format
        self.sos_token_num = sos_token_num
        self.frozen_exclude = frozen_exclude if frozen_exclude is not None else []

        if sos_token_format in ["learnable_token", "pos_embedding"]:
            self.sos_token = nn.Parameter(  # 可学的sos_token。
                torch.randn(sos_token_num, 1, self.proj.shape[0])
            )
            nn.init.normal_(self.sos_token, std=0.02)
            self.frozen_exclude.append("sos_token")
        elif sos_token_format == "cls_token":
            assert sos_token_num is None, "sos_token_num should be None when sos_token_format is cls_token."
        self._freeze(self.frozen_exclude)

    def _freeze(self, frozen_exclude: t.Sequence[str]):
        _freeze(self, frozen_exclude)

    def forward(self, features: ClipOutput, normalize: bool = False) -> ClipOutput:
        # construct clip shadow features.
        cls_token = features.cls_token[self.first_layer_idx]  # 1,n,c；CLIP主干得到的cls token。
        pix_feat = features.img_feat[self.first_layer_idx]  # n,c,h,w；CLIP主干得到的image feature。
        attn_bias = features.head_att_biases
        n, c, h, w = pix_feat.shape
        x = torch.cat(
            [cls_token, pix_feat.reshape(n, c, -1).permute(2, 0, 1)]
        )  # 1+l,n,c；恢复为transformer的输入格式。

        # construct sos token.
        if self.sos_token_format == "cls_token":
            sos_token_num = attn_bias[0].shape[2]
            sos_token = cls_token.repeat(sos_token_num, 1, 1)  # (mask数：如100, n：如16, c：如768)
        elif self.sos_token_format == "learnable_token":
            sos_token = self.sos_token.expand(-1, n, -1)
        elif self.sos_token_format == "pos_embedding":
            sos_token = self.sos_token.expand(-1, n, -1) + cls_token
        else:
            raise ValueError(f"Unknown sos_token_format: {self.sos_token_format}")
        sos_token_num = sos_token.shape[0]

        # construct attn biases.
        attn_biases = self._build_attn_biases(attn_bias, target_shape=(h, w))
        if self.cross_attn:
            for i, resblock in enumerate(self.resblocks):
                if self.cross_attn:
                    sos_token = cross_attn_layer(
                        resblock,
                        sos_token,
                        x[1:,],
                        attn_biases[i],
                    )
                    if i < len(self.resblocks) - 1:
                        x = resblock(x)
        else:
            x = torch.cat([sos_token, x], dim=0)
            for i, resblock in enumerate(self.resblocks):  # sos token在聚合语义，聚合算法与cls token一致，只是有了bias的指导。
                x = resblock(x, attn_mask=attn_biases[i])
            sos_token = x[:sos_token_num]

        sos_token = sos_token.permute(1, 0, 2)  # LND -> NLD；以下sos token与CLIP的cls token的处理是一致的，得到语义向量。

        sos_token = self.ln_post(sos_token)

        if self.proj is not None:
            sos_token = sos_token @ self.proj
        if normalize:
            sos_token = F.normalize(sos_token, dim=-1)

        features.mask_embs = sos_token
        return features

    def _build_attn_biases(self, attn_biases, target_shape):
        formatted_attn_biases = []
        for attn_bias in attn_biases:  # batch_size, num_head, num_sos, h, w：如16, 12, 100, 40, 40
            # convert it to proper format: N*num_head,L,L
            # attn_bias: [N, num_head/1, num_sos,H,W]
            n, num_head, num_sos, h, w = attn_bias.shape
            # reshape and downsample；attn bias首先缩放到与image feature相同的尺寸。
            attn_bias = downsample2d(
                attn_bias.reshape(n, num_head * num_sos, h, w),
                target_shape,
                method=self.downsample_method,  # 使用max pooling下采样，表示一块区域中，只要有感兴趣物体，就总是被选中。
            )
            attn_bias = attn_bias.reshape(n, num_head, num_sos, *target_shape)
            true_num_head = self.resblocks[0].attn.num_heads
            assert (
                num_head == 1 or num_head == true_num_head
            ), f"num_head={num_head} is not supported."
            if num_head == 1:
                attn_bias = attn_bias.repeat(1, true_num_head, 1, 1, 1)
            attn_bias = attn_bias.reshape(n * true_num_head, num_sos, -1)
            L = attn_bias.shape[-1]
            if self.cross_attn:
                # [n*num_head, num_sos, L]
                formatted_attn_biases.append(attn_bias)
            else:  # 看论文图片，cls、image feature的att与原来完全一致，sos token有bias地看image feature。
                # [n*num_head, num_sos+1+L, num_sos+1+L]
                new_attn_bias = attn_bias.new_zeros(num_sos + 1 + L, num_sos + 1 + L)
                new_attn_bias[:, :num_sos] = -100  # 不看sos token。
                new_attn_bias[torch.arange(num_sos), torch.arange(num_sos)] = 0  # sos token与自己的att bias为0。
                new_attn_bias[:num_sos, num_sos] = -100  # sos token不看cls token。
                new_attn_bias = (
                    new_attn_bias[None, ...].expand(n * true_num_head, -1, -1).clone()
                )
                new_attn_bias[..., :num_sos, -L:] = attn_bias
                formatted_attn_biases.append(new_attn_bias)

        if len(formatted_attn_biases) == 1:  # SAN实际只有一个attn bias，复制到每一层。
            formatted_attn_biases = [formatted_attn_biases[0] for _ in self.resblocks]
        return formatted_attn_biases


class SemWithMaskPooling(nn.Module):
    """使用MaskPooling捕捉语义向量。"""
    def __init__(self,
                 clip: CLIP,
                 mask_emb_aff_method: str | dict='no_aff',
                 frozen_exclude: list[str] | str = None,
                 pool_method: dict | str='up_emb_avg'):
        super().__init__()
        visual_encoder = clip.visual
        self.ln_post = visual_encoder.ln_post
        self.proj = visual_encoder.proj
        self.mask_emb_aff_method = mask_emb_aff_method
        self.pool_method = {'method': 'up_emb_avg'} if pool_method == 'up_emb_avg' else pool_method
        self.frozen_exclude = frozen_exclude if frozen_exclude is not None else []
        self._freeze(self.frozen_exclude)

    def _freeze(self, frozen_exclude: t.Sequence[str]):
        _freeze(self, frozen_exclude)

    def forward(self, clip_out: ClipOutput) -> list[torch.Tensor]:
        clip_out = self.ln_proj_patch_emb(clip_out)  # NDHW，所有patch的嵌入。
        clip_out = self.gather_patch_emb(clip_out)

        return clip_out

    def ln_proj_patch_emb(self, clip_out: ClipOutput) -> ClipOutput:
        """完成最后的LN和投影。"""
        patch_emb_h, patch_emb_w = clip_out.spacial_shape

        x = clip_out[clip_out.last_idx]  # LND，最后一层的输出。

        x = x.permute(1, 0, 2)  # NLD
        x = self.ln_post(x)  # NLD

        patch_emb = x[:, 1:, :]  # N(L-1)D，所有patch的嵌入。
        patch_emb = patch_emb @ self.proj  # 先投影再求平均，等价先求平均再投影。

        patch_emb = patch_emb.view(patch_emb.shape[0], patch_emb_h, patch_emb_w, -1).permute(0, 3, 1, 2)  # NDHW

        clip_out.patch_emb = patch_emb
        return clip_out

    @staticmethod
    def avg_emb_on_m(emb: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        # 等价于：mask_emb = (torch.sum(emb[None, :, :] * m[:, :, None], dim=1) /  # MD
        #             m.sum(dim=1, keepdim=True, dtype=emb.dtype))  # 会OOM

        # 也等价于：mask_emb = []
        # for m_ in m:
        #     mask_emb.append((emb * m_[:, None]).sum(dim=0) / m_.sum(dtype=emb.dtype))
        # mask_emb = torch.stack(mask_emb, dim=0)
        def mask_mean(e, m_):
            return (e * m_[:, None]).sum(dim=0) / m_.sum(dtype=e.dtype)

        mask_emb = torch.vmap(mask_mean, in_dims=(None, 0), out_dims=0, chunk_size=10)(emb, m)  # MD
        return mask_emb

    @staticmethod
    def einsum_avg_emb_on_m(emb: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        m = m.to(dtype=emb.dtype)
        m_area = m.sum(dim=1)  # M
        mask_emb = torch.einsum('ud,mu->md', emb, m)  # MD
        mask_emb = mask_emb / m_area[:, None]
        return mask_emb

    @staticmethod
    def _weight_mask_by_aff_bar(mask: torch.Tensor, aff_bar_out: 'DALayers.AffBarOut') -> torch.Tensor:
        # -* 取出aff_bar，变为[M, H, W]。
        # NOTE 只要用了aff_bar，默认batch_size=1。
        mask_weight = aff_bar_out.aff_bar_per_m[:, :, 1:].mean(dim=1)  # [M, 1*H, L] -> [M, L-1]
        mask_weight = mask_weight.view(mask.shape[0], *aff_bar_out.spacial_shape)

        # -* 无效mask_weight置为1.。
        invalid_aff_bar = t.cast(torch.Tensor, aff_bar_out.m_area < 0.5)  # M
        mask_weight[invalid_aff_bar, :, :] = 1.

        # -* 如果mask的形状与mask_weight不一致，将mask_weight插值到mask的形状。
        if mask_weight.shape[1:] != mask.shape[1:]:
            mask_weight = F.interpolate(mask_weight.unsqueeze(0), size=mask.shape[1:],
                                        mode='bilinear', align_corners=False).squeeze(0)

        # -* 将mask_weight与mask相乘。
        weighted_mask = mask.to(dtype=mask_weight.dtype) * mask_weight

        return weighted_mask

    def _gather_patch_emb_by_pool_method(self, idx: int, clip_out: ClipOutput) -> torch.Tensor:
        emb, m = clip_out.patch_emb[idx], clip_out.masks[idx]    # emb: DHW, m: MHW
        emb_d, mask_num = emb.shape[0], m.shape[0]

        def up_emb_avg(emb_: torch.Tensor, m_: torch.Tensor, weight_mask_by_aff_bar: bool=False) -> torch.Tensor:
            # 将emb差值到mask尺寸。
            emb_ = F.interpolate(emb_.unsqueeze(0), size=m_.shape[1:],
                                 mode='bilinear', align_corners=False).squeeze(0).flatten(1).permute(1, 0)  # (HW)D
            if weight_mask_by_aff_bar:
                m_ = self._weight_mask_by_aff_bar(m_, clip_out.aff_bar_out)
            m_ = m_.flatten(1)  # M(HW)

            return self.einsum_avg_emb_on_m(emb_, m_)  # MD

        match self.pool_method:
            case {'method': 'up_emb_avg', **other_pool_cfg}:
                return up_emb_avg(emb, m, weight_mask_by_aff_bar=other_pool_cfg.get('weight_mask_by_aff_bar', False))
            case {'method': 'area_leveled_up_emb_avg',
                  'area_property': str(area_property),
                  'prop_thresh': tuple(prop_thresh),
                  **other_pool_cfg}:
                if other_pool_cfg.get('weight_mask_by_aff_bar', False):
                    raise NotImplementedError("weight_mask_by_aff_bar is not implemented for area_leveled_up_emb_avg.")

                prop_thresh = (float('-inf'),) + prop_thresh + (float('inf'),)
                assert len(prop_thresh) - 1 == len(clip_out.leveled_patch_embs)

                sps = clip_out.sps[idx]
                leveled_patch_emb = [e[idx] for e in clip_out.leveled_patch_embs]

                mask_emb = torch.empty((mask_num, emb_d), dtype=emb.dtype, device=emb.device)  # MD
                props = (getattr(sps, area_property) / sps.img_area) ** 0.5  # M

                for i in range(len(leveled_patch_emb)):
                    valid_m = (prop_thresh[i] <= props) & (props < prop_thresh[i + 1])  # M
                    if valid_m.any():
                        mask_emb[valid_m, :] = up_emb_avg(leveled_patch_emb[i], m[valid_m, ...],
                                                          weight_mask_by_aff_bar=False)

                return mask_emb
            case {'method': 'down_mask_avg',
                  'empty_mask_roll_back': empty_mask_roll_back,
                  'downsample_method': downsample_method, **other_pool_cfg}:
                low_res_emb = emb.flatten(1).permute(1, 0)  # (HW)D

                low_res_m = clip_out.get_low_res_mask(downsample_method)[idx]  # Mhw
                valid_low_res_m = low_res_m.sum(dim=(1, 2)) > 0.5  # M
                if other_pool_cfg.get('weight_mask_by_aff_bar', False):
                    low_res_m = self._weight_mask_by_aff_bar(low_res_m, clip_out.aff_bar_out)
                low_res_m = low_res_m.flatten(1)  # M(hw)

                if valid_low_res_m.any():
                    valid_low_res_mask_emb = self.einsum_avg_emb_on_m(low_res_emb, low_res_m[valid_low_res_m, :])  # mD
                else:
                    valid_low_res_mask_emb = torch.empty((0, emb_d), dtype=emb.dtype, device=emb.device)  # D

                if valid_low_res_m.all():
                    return valid_low_res_mask_emb

                match empty_mask_roll_back:
                    case 'ones':
                        invalid_low_res_mask_emb = torch.ones(emb_d, dtype=emb.dtype, device=emb.device)  # D
                    case 'bg_emb':
                        raise NotImplementedError("empty_mask_roll_back='bg_emb' is not implemented.")
                    case 'up_emb_avg':
                        # 将emb差值到mask尺寸。
                        high_res_emb = F.interpolate(emb.unsqueeze(0), size=m.shape[1:],
                                                     mode='bilinear', align_corners=False).squeeze(0).flatten(1).permute(1, 0)  # (HW)D
                        high_res_m = m.flatten(1)  # M(HW)

                        invalid_low_res_mask_emb = self.einsum_avg_emb_on_m(high_res_emb,
                                                                            high_res_m[~valid_low_res_m, :])  # nD
                    case _:
                        raise ValueError(f"Unknown empty_mask_roll_back: {empty_mask_roll_back}")

                mask_emb = torch.empty((mask_num, emb_d), dtype=emb.dtype, device=emb.device)  # MD
                mask_emb[valid_low_res_m, :] = valid_low_res_mask_emb
                mask_emb[~valid_low_res_m, :] = invalid_low_res_mask_emb
                return mask_emb
            case _:
                raise ValueError(f"Unknown pool_method: {self.pool_method}")

    def gather_patch_emb(self, clip_out: ClipOutput) -> ClipOutput:
        mask_embs: list[torch.Tensor] = []

        for idx in range(clip_out.patch_emb.shape[0]):
            mask_emb = self._gather_patch_emb_by_pool_method(idx, clip_out)

            match self.mask_emb_aff_method:
                case 'no_aff':
                    pass
                case {'method': 'self_att',
                      'sim_type': str(sim_type),
                      'sim_scale': float() | str() as sim_scale,
                      'aff_alpha': float(aff_alpha)}:
                    mask_emb = self._mask_emb_self_att(mask_emb,
                                                       sim_type=sim_type, sim_scale=sim_scale, aff_alpha=aff_alpha)
                case _:
                    raise ValueError(f"Unknown mask_emb_aff_method: {self.mask_emb_aff_method}")

            mask_embs.append(mask_emb)

        clip_out.mask_embs = mask_embs
        return clip_out

    @staticmethod
    def _mask_emb_self_att(mask_emb: torch.Tensor,
                           sim_type: str, sim_scale: float | str, aff_alpha: float) -> torch.Tensor:
        match sim_scale:
            case 'auto':
                assert sim_type == 'dot', f"Auto sim_type is only supported for dot, not {sim_type}."
                sim_scale = mask_emb.shape[-1] ** -0.5
            case float():
                pass
            case _:
                raise ValueError(f"Unknown sim_scale: {sim_scale}")

        match sim_type:
            case 'cos':
                mask_emb_normed = F.normalize(mask_emb, p=2, dim=-1)  # MD
                sim = (mask_emb_normed @ mask_emb_normed.transpose(0, 1)) * sim_scale  # MxM
            case 'dot':
                sim = (mask_emb @ mask_emb.transpose(0, 1)) * sim_scale  # MxM
            case _:
                raise ValueError(f"Unknown sim_type: {sim_type}")
        sim = F.softmax(sim, dim=-1)  # MxM

        mask_emb_affed = sim @ mask_emb  # MxD

        return aff_alpha * mask_emb_affed + (1 - aff_alpha) * mask_emb  # MxD


class NoAttCLIPLayers(nn.Module):
    """运行CLIP的中间ResBlocks，但使用对角线的注意力偏置，即不使用注意力。"""
    def __init__(
        self,
        clip: CLIP,
        first_layer_idx: int = 0,
        frozen_exclude: list[str] | str = None,
        att_residual_weight: tuple[float, float] = (1., 1.)
    ):
        super().__init__()
        visual_encoder: VisionTransformer = clip.visual
        self.first_layer_idx = first_layer_idx  # SAN中为9。

        self.resblocks: list[ResidualAttentionBlock] = visual_encoder.transformer.resblocks[first_layer_idx:]  # CLIP剩余的resblocks，对Vit-B，是9-11层。

        self.frozen_exclude = frozen_exclude if frozen_exclude is not None else []

        self.att_residual_weight = att_residual_weight

        self._freeze(self.frozen_exclude)

    def _freeze(self, frozen_exclude: t.Sequence[str]):
        _freeze(self, frozen_exclude)

    def forward(self, clip_out: ClipOutput) -> ClipOutput:
        x = clip_out[clip_out.last_idx]

        l = x.shape[0]
        att_mask = torch.full((l, l), -100, dtype=x.dtype, device=x.device)
        att_mask.diagonal(dim1=0, dim2=1).fill_(0)  # 对角线为0，即允许每个token看到自己。
        att_mask[0, 1:].fill_(0)  # cls token可以看到所有patch。

        for i, resblock in enumerate(self.resblocks, start=clip_out.last_idx + 1):
            x = (self.att_residual_weight[0] *
                 x +
                 self.att_residual_weight[1] *
                 resblock.ls_1(resblock.attention(q_x=resblock.ln_1(x), attn_mask=att_mask)))
            x = x + resblock.ls_2(resblock.mlp(resblock.ln_2(x)))
            clip_out.save(i, x)

        return clip_out


class SCLIPLayers(nn.Module):
    """运行CLIP的中间ResBlocks，但使用对角线的注意力偏置，即不使用注意力。"""
    def __init__(
        self,
        clip: CLIP,
        first_layer_idx: int=-1,
        frozen_exclude: list[str] | str=None,
        custom_att_factor: float=None,
        att_residual_weight: tuple[float, float] = (1., 1.),
        norm_att: bool=False,
        avg_att: bool=False,
        mask_clip_balancer: tuple[float, float] = (0., 1.)
    ):
        super().__init__()
        visual_encoder: VisionTransformer = clip.visual

        self.first_layer_idx = first_layer_idx  # SCLIP中为-1。
        self.resblocks: list[ResidualAttentionBlock] = visual_encoder.transformer.resblocks[first_layer_idx:]

        self.frozen_exclude = frozen_exclude if frozen_exclude is not None else []
        self._freeze(self.frozen_exclude)

        self.custom_att_factor = custom_att_factor
        self.att_residual_weight = att_residual_weight
        self.norm_att = norm_att
        self.avg_att = avg_att
        self.mask_clip_balancer = mask_clip_balancer

    def _freeze(self, frozen_exclude: t.Sequence[str]):
        _freeze(self, frozen_exclude)

    def forward(self, clip_out: ClipOutput) -> ClipOutput:
        att_biases = clip_out.head_att_biases
        if att_biases is None:
            att_biases = [None] * len(self.resblocks)
        else:
            assert len(att_biases) == len(self.resblocks), f"{len(att_biases)=}≠{len(self.resblocks)=}"

        x = clip_out[clip_out.last_idx]

        for i, (resblock, att_bias) in enumerate(zip(self.resblocks, att_biases, strict=True),
                                                 start=clip_out.last_idx + 1):
            x = (self.att_residual_weight[0] * x +
                 self.att_residual_weight[1] * self.custom_attn(resblock, x,
                                                                att_bias=att_bias,
                                                                custom_att_factor=self.custom_att_factor,
                                                                csa=True, norm_att=self.norm_att, avg_att=self.avg_att,
                                                                mask_clip_balancer=self.mask_clip_balancer))
            x = x + resblock.mlp(resblock.ln_2(x))
            clip_out.save(i, x)

        return clip_out

    @staticmethod
    def custom_attn(resblock: ResidualAttentionBlock, x: torch.Tensor,
                    att_bias: torch.Tensor=None, custom_att_factor: float=None,
                    return_attn: bool | str=False, with_attn: bool=False,
                    csa: bool=False, norm_att: bool=False, avg_att: bool=False,
                    mask_clip_balancer: tuple[float, float] = (0., 1.)) \
            -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        attn_layer = resblock.attn
        num_heads = attn_layer.num_heads
        _, bsz, embed_dim = x.size()
        head_dim = embed_dim // num_heads
        scale = head_dim ** -0.5 if custom_att_factor is None else custom_att_factor ** -1

        x = resblock.ln_1(x)

        q, k, v = F.linear(x, attn_layer.in_proj_weight, attn_layer.in_proj_bias).chunk(3, dim=-1)  # [L, N, D]
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)  # [N*h, L, D/h]
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        att_bias = _canonical_mask(
            mask=att_bias,
            mask_name="att_bias",
            other_type=None,
            other_name="",
            target_type=q.dtype,
            check_other=False,
        )

        if csa:
            if att_bias is not None:
                q_attn = torch.baddbmm(att_bias, q * scale, q.transpose(1, 2))  # [N*h, L, L]
                k_attn = torch.baddbmm(att_bias, k * scale, k.transpose(1, 2))
            else:
                q_attn = torch.bmm(q * scale, q.transpose(1, 2))  # [N*h, L, L]
                k_attn = torch.bmm(k * scale, k.transpose(1, 2))
            attn_weights = F.softmax(q_attn, dim=-1) + F.softmax(k_attn, dim=-1)
            if norm_att:
                attn_weights = attn_weights / 2
        else:
            if att_bias is not None:
                attn_weights = torch.baddbmm(att_bias, q * scale, k.transpose(1, 2))
            else:
                attn_weights = torch.bmm(q * scale, k.transpose(1, 2))
            attn_weights = F.softmax(attn_weights, dim=-1)

        if avg_att:
            attn_weights = attn_weights.view(bsz, -1, *attn_weights.shape[1:]).mean(dim=1, keepdim=True)  # [N, 1, L, L]
            attn_weights = attn_weights.expand(-1, num_heads, -1, -1).view(-1, *attn_weights.shape[2:])  # [N*h, L, L]

        def out_proj(out: torch.Tensor) -> torch.Tensor:
            out = out.transpose(0, 1).contiguous().view(-1, bsz, embed_dim)  # [L, N, D]
            return attn_layer.out_proj(out)

        match return_attn:
            case False:
                pass
            case True | 'proj_v':
                attn_output = out_proj(v).view(-1, bsz * num_heads, head_dim).transpose(0, 1)  # [N*h, L, D/h]
                return attn_output, attn_weights
            case 'v':
                return v, attn_weights
            case _:
                raise ValueError(f"Unknown return_attn: {return_attn}")

        attn_output = torch.bmm(attn_weights, v)  # [N*h, L, D/h]
        attn_output = mask_clip_balancer[0] * v + mask_clip_balancer[1] * attn_output
        attn_output = out_proj(attn_output)

        if with_attn:
            return attn_output, attn_weights

        return attn_output


class DALayers(nn.Module):
    """Discrimination Alignment Layers"""
    def __init__(
        self,
        clip: CLIP,
        first_layer_idx: int=-1,
        frozen_exclude: list[str] | str=None,
        custom_att_factor: float=None,
        norm_att: bool=False,
        att_residual_weight: tuple[float, float] = (1., 1.),
        proj_v_first: bool=True,
        mask_downsample_method: str='nearest',
        aff_method: dict=None,
        aff_bar_method: dict=None,
        mask_clip_balancer: tuple[float, float] = (0., 1.),
        save_aff_bar_out: bool=False
    ):
        super().__init__()
        visual_encoder: VisionTransformer = clip.visual

        self.first_layer_idx = first_layer_idx  # SCLIP中为-1。
        self.resblocks: list[ResidualAttentionBlock] = visual_encoder.transformer.resblocks[first_layer_idx:]

        self.frozen_exclude = frozen_exclude if frozen_exclude is not None else []
        self._freeze(self.frozen_exclude)

        self.custom_att_factor = custom_att_factor
        self.norm_att = norm_att
        self.att_residual_weight = att_residual_weight
        self.proj_v_first = proj_v_first
        self.mask_downsample_method = mask_downsample_method
        self.aff_method = aff_method
        if aff_bar_method is None:
            aff_bar_method = {'method': 'avg', 'can_see_cls_token': True}
        self.aff_bar_method = aff_bar_method
        self.mask_clip_balancer = mask_clip_balancer
        self.save_aff_bar_out = save_aff_bar_out

    def _freeze(self, frozen_exclude: t.Sequence[str]):
        _freeze(self, frozen_exclude)

    def forward(self, clip_out: ClipOutput) -> ClipOutput:
        att_biases = clip_out.head_att_biases
        if att_biases is None:
            att_biases = [None] * len(self.resblocks)
        else:
            assert len(att_biases) == len(self.resblocks), f"{len(att_biases)=}≠{len(self.resblocks)=}"

        assert len(clip_out.masks) == 1, "DALayers only supports one mask and must have a mask."

        x = clip_out[clip_out.last_idx]
        low_res_mask = clip_out.get_low_res_mask(self.mask_downsample_method)[0]

        for i, (resblock, att_bias) in enumerate(zip(self.resblocks, att_biases, strict=True),
                                                 start=clip_out.last_idx + 1):
            x = (self.att_residual_weight[0] * x +
                 self.att_residual_weight[1] * self.dal(resblock, x, low_res_mask, att_bias, clip_out))
            x = x + resblock.mlp(resblock.ln_2(x))
            clip_out.save(i, x)

        return clip_out

    def dal(self, resblock: ResidualAttentionBlock, x: torch.Tensor,
            low_res_mask: torch.Tensor, att_bias: torch.Tensor,
            clip_out: ClipOutput) -> torch.Tensor:
        """执行DAL。

        Args:
            resblock: CLIP最后一层。
            x: [L, N, D] clip中间特征。
            low_res_mask: [M,H,W] 与x的分辨率一致的掩码。
            att_bias: [L, L]多头注意力偏置。
            clip_out: CLIP输出。

        Returns:
            [L, N, D]经过辨别力对齐的clip特征。
        """
        _, bsz, embed_dim = x.size()

        # -* 获取线性变换后的V，以及注意力权重。
        out_v, attn_weights = SCLIPLayers.custom_attn(resblock, x,  # out_v: # [N*h, L, D/h]; attn_weights: [N*h, L, L]
                                                      att_bias=att_bias,
                                                      custom_att_factor=self.custom_att_factor,
                                                      return_attn='proj_v' if self.proj_v_first else 'v',
                                                      csa=True, norm_att=self.norm_att)

        # -* 将注意力权重转为aff。
        aff = self.get_aff(attn_weights, bsz, self.aff_method)

        # -* 获取辨别力对齐的特征。
        aff_bar_out = self.get_aff_bar(low_res_mask, aff, self.aff_bar_method)  # [N*h, L, L]
        if self.save_aff_bar_out:
            clip_out.aff_bar_out = aff_bar_out
        dal_v = aff_bar_out.aff_bar @ out_v  # [N*h, L, D/h]

        # -* 与线性变换后v（MaskCLIP的v）加权平均。
        dal_v = self.mask_clip_balancer[0] * out_v + self.mask_clip_balancer[1] * dal_v

        # -* 改变为标准形状。
        dal_v = dal_v.transpose(0, 1).contiguous().view(-1, bsz, embed_dim)  # [L, N, D]

        # -* 若后投影，进行投影。
        if not self.proj_v_first:
            dal_v = resblock.attn.out_proj(dal_v)

        return dal_v

    @staticmethod
    def get_aff(attn_weights: torch.Tensor, bsz: int, aff_method: dict=None) -> torch.Tensor:
        """从原始多头注意力权重中获取aff。

        Args:
            attn_weights: [N*h, L, L]多头注意力权重。
            bsz: batch size。
            aff_method: 从attn_weights中获取aff的方法。

        Returns:
            [N*h, L, L]aff。
        """
        match aff_method:
            case None | {'method': 'identity'}:
                aff = attn_weights
            case {'method': 'avg_head'}:
                aff = attn_weights.view(bsz, -1, *attn_weights.shape[1:])  # [N, h, L, L]
                head_num = aff.shape[1]
                aff = aff.mean(dim=1, keepdim=True).expand(-1, head_num, -1, -1).view(-1, *attn_weights.shape[1:])  # [N*h, L, L]
            case _:
                raise ValueError(f"Unknown aff_method: {aff_method}")

        return aff

    @dataclass
    class AffBarOut(object):
        spacial_shape: tuple[int, int] = None
        aff_bar: torch.Tensor = None
        aff_bar_per_m: torch.Tensor = None
        m_area: torch.Tensor = None

    @staticmethod
    def get_aff_bar(low_res_mask: torch.Tensor, aff: torch.Tensor, aff_bar_method: dict) -> AffBarOut:
        """从aff中获取aff_bar。

        Args:
            low_res_mask: [M,H,W]低分辨率float32掩码。
            aff: [N*h, L, L]aff。
            aff_bar_method: 从aff中获取aff_bar的方法。

        Returns:
            [N*h, L, L]aff_bar。
        """
        match aff_bar_method:
            case {'method': 'identity'}:
                aff_bar = aff
                ret = DALayers.AffBarOut(aff_bar=aff_bar)
            case {'method': 'avg', 'avg_cls_token': bool(avg_cls_token)}:
                # -* 拉平掩码。
                m_flatten = low_res_mask.flatten(1)  # [M,H*W] = [M, L-1]
                m_flatten = F.pad(m_flatten, (1, 0), value=1. if avg_cls_token else 0.)  # [M, L]

                # -* 每个掩码的面积、每个位置的掩码数量。
                m_area = torch.einsum('ml->m', m_flatten)  # [M]
                m_count = torch.einsum('ml->l', m_flatten)  # [L]

                # -* 每个掩码的平均aff。
                aff_bar_per_m = torch.einsum('ml,hlk->mhk', m_flatten, aff) / (m_area[:, None, None] + 1e-3)  # [M, N*H, L]

                # -* 将掩码的平均aff，放回该掩码对应位置。
                aff_bar = torch.einsum('mhk,ml->hlk', aff_bar_per_m, m_flatten)  # [N*H, L, L]

                # -* 若一个位置对应多个mask，应当取平均。
                aff_bar = aff_bar / (m_count[None, :, None] + 1e-3)

                ret = DALayers.AffBarOut(spacial_shape=low_res_mask.shape[1:],
                                         aff_bar=aff_bar, aff_bar_per_m=aff_bar_per_m, m_area=m_area)
            case _:
                raise ValueError(f"Unknown aff_bar_method: {aff_bar_method}")

        return ret
