#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/12/18 21:00
@File    : clip_extractor.py
@Software: PyCharm
@Desc    : 
"""
import typing as t
import warnings

import torch
import torch.nn.functional as F
from alchemy_cat.alg import slide_inference
from open_clip import CLIP
from torch import nn

from libs.clip.visual import FeatureExtractor, RecWithAttnbiasHead, ClipOutput, NoAttCLIPLayers, SemWithMaskPooling, \
    SCLIPLayers, DALayers
from libs.sam import SamAnns

__all__ = ['AttPoolCLIPWithMaskLikeSAN', 'MaskCLIP', 'SCLIP']


class AttPoolCLIPWithMaskLikeSAN(nn.Module):

    def __init__(self,
                 clip: CLIP,
                 head_layer_idx: int,
                 mask2bias: dict[str, ...],
                 feature_extractor_frozen_exclude: t.Sequence[str]=None,
                 head_frozen_exclude: t.Sequence[str]=None,
                 bias_downsample_method: str='max',
                 ):
        super().__init__()

        self.mask2bias = mask2bias

        self.clip_visual_extractor = FeatureExtractor(
            clip,
            last_layer_idx=head_layer_idx,
            frozen_exclude=feature_extractor_frozen_exclude,
        )

        self.clip_rec_head = RecWithAttnbiasHead(  # CLIP语义聚合器。
            clip,
            first_layer_idx=head_layer_idx,
            frozen_exclude=head_frozen_exclude,
            sos_token_format='cls_token',
            sos_token_num=None,
            cross_attn=False,
            downsample_method=bias_downsample_method,
        )

        self.patch_size = clip.visual.patch_size
        self.num_heads: int = self.clip_rec_head.resblocks[0].attn.num_heads

    def att_pool(self, imgs: torch.Tensor, attn_biases: torch.Tensor) -> torch.Tensor:
        """基于img和attn_biases，为每一个attn_bias生成一个mask embedding。

        Args:
            imgs: (batch_size, 3, h, w)的图像。
            attn_biases: (batch_size, num_head, num_mask, h, w)的注意力偏置。若attn_biases的(h, w)不必和imgs相同，则
                按照bias_downsample_method将attn_biases下采样到和imgs相同的(h, w)。

        Returns:
            (batch_size, num_mask, feature_dim)的mask embedding。
        """
        assert imgs.shape[2] % self.patch_size[0] == 0 and imgs.shape[3] % self.patch_size[1] == 0, \
            f"Image shape {imgs.shape} is not divisible by patch size {self.patch_size}."

        clip_output = ClipOutput((imgs.shape[2] // self.patch_size[0], imgs.shape[3] // self.patch_size[1]),
                                 imgs=imgs, head_att_biases=[attn_biases])

        clip_out: ClipOutput = self.clip_visual_extractor(clip_output)

        clip_out: ClipOutput = self.clip_rec_head(clip_out, normalize=False) # [B,N,C]：如1,100,512

        return clip_out.mask_embs

    def forward(self, imgs: torch.Tensor, masks: list[torch.Tensor]) -> torch.Tensor:
        """基于img和masks，为每一个mask生成一个mask embedding。
        函数本体主要处理将masks转为attn_biases。

        Args:
            imgs: (batch_size, 3, h, w)的图像。目前batch_size只支持1。
            masks: batch_size个(num_mask, h, w)的掩码。

        Returns:
            (batch_size, num_mask, feature_dim)的mask embedding。
        """
        warnings.warn("This class is deprecated.", DeprecationWarning)
        assert imgs.shape[0] == 1, "Only support batch size 1."
        assert len(masks) == 1, "Only support batch size 1."

        attn_biases: list[torch.Tensor] = []
        for mask in masks:
            match self.mask2bias:
                case {'method': 'binary', 'fg_bias': float(fg_bias), 'bg_bias': float(other_bias)}:
                    attn_bias = self.mask2bias_by_binary(mask, fg_bias, other_bias)
                case _:
                    raise NotImplementedError(f"Unknown mask2bias method {self.mask2bias['method']}.")
            attn_bias = attn_bias.expand(self.num_heads, -1, -1, -1)
            attn_biases.append(attn_bias)

        mask_embs = self.att_pool(imgs, torch.stack(attn_biases, dim=0))

        return mask_embs

    @staticmethod
    def mask2bias_by_binary(masks: torch.Tensor, fg_bias: float, bg_bias: float) -> torch.Tensor:
        """将mask转为attn_bias。

        Args:
            masks: (num_mask, h, w)的布尔掩码。
            fg_bias: 前景的attn_bias。
            bg_bias: 背景的attn_bias。

        Returns:
            (num_head, num_mask, h, w)的attn_bias。
        """
        attn_bias = torch.where(masks, fg_bias, bg_bias)  # TODO 对齐CLIP的推理类型。
        return attn_bias


class MaskCLIP(nn.Module):

    def __init__(self,
                 clip: CLIP,
                 head_layer_idx: int,
                 backbone_frozen_exclude: t.Sequence[str]=None,
                 head_frozen_exclude: t.Sequence[str]=None,
                 infer_method: dict | str='one-forward',
                 backbone_att_bias_method: dict | str='no_att',
                 head_att_bias_method: dict | str = 'no_att',
                 pool_method: dict | str = 'up_emb_avg',
                 mask_emb_aff_method: str | dict='no_aff',
                 head_att_residual_weight: tuple[float, float] = (1., 1.)
                 ):
        super().__init__()

        self.backbone = FeatureExtractor(
            clip,
            last_layer_idx=head_layer_idx,
            frozen_exclude=backbone_frozen_exclude,
        )

        self.head = NoAttCLIPLayers(
            clip,
            first_layer_idx=head_layer_idx,
            frozen_exclude=head_frozen_exclude,
            att_residual_weight=head_att_residual_weight
        )

        self.mask_pooler = SemWithMaskPooling(clip, mask_emb_aff_method=mask_emb_aff_method, pool_method=pool_method)

        self.infer_method = infer_method
        self.backbone_att_bias_method = backbone_att_bias_method
        self.head_att_bias_method = head_att_bias_method

        self.patch_size = clip.visual.patch_size
        self.emb_dim = self.mask_pooler.proj.shape[1]

    @classmethod
    def sclip(cls,
              clip: CLIP,
              head_layer_idx: int,
              backbone_frozen_exclude: t.Sequence[str]=None,
              head_frozen_exclude: t.Sequence[str]=None,
              infer_method: dict | str='one-forward',
              backbone_att_bias_method: dict | str='no_att',
              head_att_bias_method: dict | str='no_att',
              pool_method: dict | str = 'up_emb_avg',
              mask_emb_aff_method: str | dict='no_aff',
              sclip_custom_att_factor: float=None,
              head_att_residual_weight: tuple[float, float] = (1., 1.),
              norm_att: bool=False,
              avg_att: bool=False,
              mask_clip_balancer: tuple[float, float] = (0., 1.)
              ) -> 'MaskCLIP':
        """生成SCLIP。"""
        sclip = cls(clip=clip,
                    head_layer_idx=head_layer_idx,
                    backbone_frozen_exclude=backbone_frozen_exclude,
                    head_frozen_exclude=head_frozen_exclude,
                    infer_method=infer_method,
                    backbone_att_bias_method=backbone_att_bias_method,
                    pool_method=pool_method,
                    head_att_bias_method=head_att_bias_method,
                    mask_emb_aff_method=mask_emb_aff_method)

        sclip.head = SCLIPLayers(
            clip,
            first_layer_idx=head_layer_idx,
            frozen_exclude=head_frozen_exclude,
            custom_att_factor=sclip_custom_att_factor,
            att_residual_weight=head_att_residual_weight,
            norm_att=norm_att,
            avg_att=avg_att,
            mask_clip_balancer=mask_clip_balancer
        )

        return sclip

    @classmethod
    def da_clip(cls,
                clip: CLIP,
                head_layer_idx: int,
                backbone_frozen_exclude: t.Sequence[str] = None,
                head_frozen_exclude: t.Sequence[str] = None,
                infer_method: dict | str = 'one-forward',
                backbone_att_bias_method: dict | str = 'no_att',
                head_att_bias_method: dict | str = 'no_att',
                pool_method: dict | str = 'up_emb_avg',
                mask_emb_aff_method: str | dict = 'no_aff',
                sclip_custom_att_factor: float = None,
                norm_att: bool=False,
                head_att_residual_weight: tuple[float, float] = (1., 1.),
                proj_v_first: bool = True,
                aff_method: dict=None,
                aff_bar_method: dict=None,
                mask_clip_balancer: tuple[float, float] = (0., 1.),
                save_aff_bar_out: bool=False
                ) -> 'MaskCLIP':
        """生成DAL。"""
        da_clip = cls(clip=clip,
                      head_layer_idx=head_layer_idx,
                      backbone_frozen_exclude=backbone_frozen_exclude,
                      head_frozen_exclude=head_frozen_exclude,
                      infer_method=infer_method,
                      backbone_att_bias_method=backbone_att_bias_method,
                      head_att_bias_method=head_att_bias_method,
                      pool_method=pool_method,
                      mask_emb_aff_method=mask_emb_aff_method)

        da_clip.head = DALayers(
            clip,
            first_layer_idx=head_layer_idx,
            frozen_exclude=head_frozen_exclude,
            custom_att_factor=sclip_custom_att_factor,
            norm_att=norm_att,
            att_residual_weight=head_att_residual_weight,
            proj_v_first=proj_v_first,
            mask_downsample_method=head_att_bias_method['downsample_method'],
            aff_method=aff_method,
            aff_bar_method=aff_bar_method,
            mask_clip_balancer=mask_clip_balancer,
            save_aff_bar_out=save_aff_bar_out
        )

        return da_clip

    @classmethod
    def mask_pooling_da_clip(cls,
                             clip: CLIP,
                             head_layer_idx: int,
                             backbone_frozen_exclude: t.Sequence[str] = None,
                             head_frozen_exclude: t.Sequence[str] = None,
                             infer_method: dict | str = 'one-forward',
                             backbone_att_bias_method: dict | str = 'no_att',
                             head_att_bias_method: dict | str = 'no_att',
                             pool_method: dict | str = 'up_emb_avg',
                             mask_emb_aff_method: str | dict = 'no_aff',
                             sclip_custom_att_factor: float = None,
                             head_att_residual_weight: tuple[float, float] = (1., 1.)
                             ) -> 'MaskCLIP':
        pool_method: dict = {'method': 'up_emb_avg'} if pool_method == 'up_emb_avg' else dict(pool_method)
        pool_method['weight_mask_by_aff_bar'] = True

        mask_pooling_da_clip = cls.da_clip(clip=clip,
                                           head_layer_idx=head_layer_idx,
                                           backbone_frozen_exclude=backbone_frozen_exclude,
                                           head_frozen_exclude=head_frozen_exclude,
                                           infer_method=infer_method,
                                           backbone_att_bias_method=backbone_att_bias_method,
                                           head_att_bias_method=head_att_bias_method,
                                           pool_method=pool_method,
                                           mask_emb_aff_method=mask_emb_aff_method,
                                           sclip_custom_att_factor=sclip_custom_att_factor,
                                           norm_att=True,
                                           head_att_residual_weight=head_att_residual_weight,
                                           proj_v_first=False,
                                           aff_method={'method': 'avg_head'},
                                           aff_bar_method={'method': 'avg', 'avg_cls_token': False},
                                           mask_clip_balancer=(1., 0.),
                                           save_aff_bar_out=True
                                           )

        return mask_pooling_da_clip

    @staticmethod
    def get_mask2mask_att_bias(clip_out: ClipOutput,
                               downsample_method: str,
                               fg_bias: float, bg_bias: float,
                               can_see_cls_tok: bool = True,
                               always_can_see_self: bool=False) -> torch.Tensor:
        assert clip_out.masks[0].dtype == torch.bool

        # -* 生成低分辨率掩码。
        low_res_mask = clip_out.get_low_res_mask(downsample_method)[0]  # [M,h,w]

        # -* 生成低分辨率掩码的掩码间注意力偏置。
        low_res_mask = low_res_mask.flatten(1)  # [M,h*w]
        mask2mask = torch.einsum('mi,mj->ij', low_res_mask, low_res_mask)  # [h*w,h*w]

        mask2mask_att_bias = torch.where(t.cast(torch.Tensor, mask2mask > 0.5), fg_bias, bg_bias)  # [h*w,h*w]

        # -* pad cls_token的注意力偏置。
        mask2mask_att_bias = F.pad(mask2mask_att_bias, (1, 0, 1, 0), value=0.)  # [h*w+1,h*w+1]
        if not can_see_cls_tok:
            mask2mask_att_bias[1:, 0] = bg_bias

        # -* 设置自身可见性。
        if always_can_see_self:
            mask2mask_att_bias.fill_diagonal_(fg_bias)

        return mask2mask_att_bias

    @classmethod
    def get_att_bias(cls,
                     att_bias_method: dict | str, att_bias_num: int,
                     clip_out: ClipOutput) -> list[torch.Tensor] | None:
        match att_bias_method:
            case 'no_att':
                att_biases = None
            case {'method': 'binary_accord_mask',
                  'downsample_method': str(downsample_method),
                  'fg_bias': float(fg_bias), 'bg_bias': float(bg_bias), 'at': list() | tuple() | int() as at,
                  'can_see_cls_tok': bool(can_see_cls_tok), **others}:
                always_can_see_self = others.get('always_can_see_self', False)
                assert len(clip_out.masks) == 1, "Currently only support batch size 1."  # TODO 支持batch size > 1。
                att_bias = cls.get_mask2mask_att_bias(clip_out,
                                                      downsample_method=downsample_method,
                                                      fg_bias=fg_bias, bg_bias=bg_bias,
                                                      can_see_cls_tok=can_see_cls_tok,
                                                      always_can_see_self=always_can_see_self
                                                      ).to(dtype=clip_out.imgs.dtype)
                at = (True,) * at if isinstance(at, int) else at
                assert (pad := att_bias_num - len(at)) >= 0
                att_biases = [None] * pad + [att_bias if elem else None for elem in at]
            case _:
                raise NotImplementedError(f"Unknown backbone_att_bias method {att_bias_method}.")

        return att_biases

    def encode_patch_emb(self, imgs: torch.Tensor, masks: list[torch.Tensor] | torch.Tensor,
                         sps: list[SamAnns]) -> ClipOutput:
        """基于img，为每一个patch生成一个patch embedding。

        Args:
            imgs: (batch_size, 3, h, w)的图像。
            masks: batch_size个(num_mask, h, w)的掩码。
            sps: batch_size个SamAnns。

        Returns:
            (batch_size, feature_dim, h, w)的patch embedding。
        """
        if torch.is_tensor(masks):  # slide inference时，masks为(num_mask, h, w)的掩码。
            masks = [masks]

        assert imgs.shape[2] % self.patch_size[0] == 0 and imgs.shape[3] % self.patch_size[1] == 0, \
            f"Image shape {imgs.shape} is not divisible by patch size {self.patch_size}."
        low_res_size = imgs.shape[2] // self.patch_size[0], imgs.shape[3] // self.patch_size[1]
        clip_out = ClipOutput(low_res_size, imgs=imgs, masks=masks, sps=sps)

        clip_out.backbone_att_biases = self.get_att_bias(self.backbone_att_bias_method, len(self.backbone.resblocks),
                                                         clip_out)
        clip_out.head_att_biases = self.get_att_bias(self.head_att_bias_method, len(self.head.resblocks),
                                                     clip_out)

        clip_out: ClipOutput = self.backbone(clip_out)

        clip_out: ClipOutput = self.head(clip_out)

        clip_out: ClipOutput = self.mask_pooler.ln_proj_patch_emb(clip_out)

        return clip_out

    def forward(self, imgs: torch.Tensor, masks: list[torch.Tensor], sps: list[SamAnns]) -> list[torch.Tensor]:
        """基于img和masks，为每一个mask生成一个mask embedding。

        Args:
            imgs: (batch_size, 3, h, w)的图像。
            masks: batch_size个(num_mask, h, w)的布尔掩码。
            sps: batch_size个SamAnns.

        Returns:
            batch_size个(num_mask, feature_dim)的mask embedding。
        """
        match self.infer_method:
            case 'one-forward':
                clip_out = self.encode_patch_emb(imgs, masks, sps)  # NDHW，所有patch的嵌入。
                mask_embs = self.mask_pooler.gather_patch_emb(clip_out).mask_embs
            case {'method': 'slide',
                  'stride': stride, 'win_size': win_size, 'with_global': float(with_global), **others}:
                level_cumsum: bool = others.get('level_cumsum', False)

                assert len(masks) == 1, "Currently only support batch size 1."  # TODO 支持batch size > 1。
                mask = F.interpolate(masks[0].unsqueeze(0).to(torch.uint8), size=imgs.shape[2:],
                                     mode='nearest-exact').squeeze(0).to(torch.bool)
                local_patch_emb = slide_inference(imgs, mask,
                                                  model=lambda i, m: self.encode_patch_emb(i, m, sps).patch_emb,
                                                  num_class=self.emb_dim,
                                                  window_sizes=win_size, strides=stride,
                                                  pad=False, align_corners=False,
                                                  win_size_checker=lambda x: x % self.patch_size[0] == 0)
                if with_global > 0.:
                    global_patch_emb = self.encode_patch_emb(imgs, masks, sps).patch_emb  # NDHW，所有patch的嵌入。
                    global_patch_emb = F.interpolate(global_patch_emb, size=local_patch_emb.shape[2:],
                                                     mode='bilinear', align_corners=False)
                    patch_emb = with_global * global_patch_emb + (1 - with_global) * local_patch_emb
                else:
                    patch_emb = local_patch_emb
                    global_patch_emb = None

                if level_cumsum:
                    leveled_patch_embs = (local_patch_emb, patch_emb)
                else:
                    leveled_patch_embs = (local_patch_emb, global_patch_emb)

                mask_embs = self.mask_pooler.gather_patch_emb(ClipOutput(patch_emb=patch_emb,
                                                                         masks=masks,
                                                                         leveled_patch_embs=
                                                                         t.cast(tuple[torch.Tensor],
                                                                                leveled_patch_embs),
                                                                         sps=sps)).mask_embs
            case {'method': 'multi_slide',
                  'stride_factor': stride_factor, 'win_sizes': win_sizes,
                  'with_global': bool(with_global), 'momentum': float(momentum),
                  **others}:
                level_cumsum: bool = others.get('level_cumsum', False)

                assert len(masks) == 1, "Currently only support batch size 1."  # TODO 支持batch size > 1。

                # -* 预处理Mask。
                mask = F.interpolate(masks[0].unsqueeze(0).to(torch.uint8), size=imgs.shape[2:],
                                     mode='nearest-exact').squeeze(0).to(torch.bool)

                # -* 多级滑窗。
                patch_embs = []
                for win_size in win_sizes:
                    stride = win_size // stride_factor
                    local_patch_emb = slide_inference(imgs, mask,
                                                      model=lambda i, m: self.encode_patch_emb(i, m, sps).patch_emb,
                                                      num_class=self.emb_dim,
                                                      window_sizes=win_size, strides=stride,
                                                      pad=False, align_corners=False,
                                                      win_size_checker=lambda x: x % self.patch_size[0] == 0)
                    patch_embs.append(local_patch_emb)

                if with_global:
                    global_patch_emb = self.encode_patch_emb(imgs, masks, sps).patch_emb  # NDHW，所有patch的嵌入。
                    global_patch_emb = F.interpolate(global_patch_emb, size=patch_embs[0].shape[2:],
                                                     mode='bilinear', align_corners=False)
                    patch_embs.append(global_patch_emb)
                else:
                    pass

                patch_emb = None
                leveled_patch_embs = ()
                for e in patch_embs:
                    patch_emb = e if patch_emb is None else momentum * e + (1 - momentum) * patch_emb
                    leveled_patch_embs += (patch_emb,) if level_cumsum else (e,)

                mask_embs = self.mask_pooler.gather_patch_emb(ClipOutput(patch_emb=patch_emb,
                                                                         masks=masks,
                                                                         leveled_patch_embs=
                                                                         t.cast(tuple[torch.Tensor],
                                                                                leveled_patch_embs),
                                                                         sps=sps)).mask_embs
            case _:
                raise NotImplementedError(f"Unknown infer method {self.infer_method}.")

        return mask_embs


class SCLIP(nn.Module):

    def __init__(self,
                 clip: CLIP,
                 head_layer_idx: int=-1,
                 feature_extractor_frozen_exclude: t.Sequence[str]=None,
                 head_frozen_exclude: t.Sequence[str]=None,
                 ):
        super().__init__()

        self.clip_visual_extractor = FeatureExtractor(
            clip,
            last_layer_idx=head_layer_idx,
            frozen_exclude=feature_extractor_frozen_exclude,
        )

        self.sclip_head = SCLIPLayers(
            clip,
            first_layer_idx=head_layer_idx,
            frozen_exclude=head_frozen_exclude,
        )

        self.mask_pooling = SemWithMaskPooling(clip)

        self.patch_size = clip.visual.patch_size

    def forward(self, imgs: torch.Tensor, masks: list[torch.Tensor], sps: list[SamAnns]) -> list[torch.Tensor]:
        """基于img和masks，为每一个mask生成一个mask embedding。

        Args:
            imgs: (batch_size, 3, h, w)的图像。
            masks: batch_size个(num_mask, h, w)的掩码。
            sps: batch_size个SamAnns.

        Returns:
            batch_size个(num_mask, feature_dim)的mask embedding。
        """
        clip_out = ClipOutput((imgs.shape[2] // self.patch_size[0], imgs.shape[3] // self.patch_size[1]),
                              imgs=imgs, masks=masks, sps=sps)

        clip_out: ClipOutput = self.clip_visual_extractor(clip_out)

        clip_out: ClipOutput = self.sclip_head(clip_out)

        clip_out: ClipOutput = self.mask_pooling(clip_out)

        return clip_out.mask_embs
