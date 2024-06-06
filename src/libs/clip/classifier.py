# COPY: SAN/san/model/clip_utils/classifier.py
from typing import List

import torch
from open_clip.model import CLIP
from open_clip.tokenizer import HFTokenizer, SimpleTokenizer
from torch import nn
from torch.nn import functional as F

from libs.classifier.typing import SPEmbClassifier
from .utils import get_labelset_from_dataset


class PredefinedOvClassifier(nn.Module):  # 将CLIP除了Visual部分的模型（模块、参数、buffer）拷贝到这里——也就是文本编码器部分。
    def __init__(
            self,
            clip_model: CLIP,
            tokenizer: SimpleTokenizer | HFTokenizer,
            cache_feature: bool = True,
            templates: List[str] = ("a photo of {}",),
            vocabulary: list[str] = None,
            dataset_name: str=None
    ):
        # copy the clip model to this module
        super().__init__()
        for name, child in clip_model.named_children():
            if "visual" not in name:  # ('transformer', 'token_embedding', 'ln_final')
                self.add_module(name, child)
        for name, param in clip_model.named_parameters(recurse=False):  # ('positional_embedding', 'text_projection', 'logit_scale')
            self.register_parameter(name, param)
        for name, buffer in clip_model.named_buffers(recurse=False):  # ('attn_mask',)
            self.register_buffer(name, buffer)

        self.tokenizer = tokenizer

        self.templates = templates
        self._freeze()

        self.cache_feature = cache_feature
        if self.cache_feature:
            self.cache = {}

        # * 初始化时，根据预设的词汇组，提前缓存好类别向量。
        assert vocabulary is None or dataset_name is None
        self.vocabulary = vocabulary
        self.dataset_name = dataset_name

    def classify_samq_emb_cal(self, emb: torch.Tensor, _: str) -> torch.Tensor:
        """emb分类器的输入数据处理。"""
        return self(emb)

    @classmethod
    def samq_emb_classifier(cls,
                            clip_model: CLIP,
                            tokenizer: SimpleTokenizer | HFTokenizer,
                            cache_feature: bool = True,
                            templates: List[str] = ("a photo of {}",),
                            vocabulary: list[str] = None,
                            dataset_name: str = None) -> SPEmbClassifier:
        """得到classify_samq_emb中的分类器。"""
        classifier = cls(clip_model=clip_model,
                         tokenizer=tokenizer,
                         cache_feature=cache_feature,
                         templates=templates,
                         vocabulary=vocabulary,
                         dataset_name=dataset_name).to('cuda')

        return classifier.classify_samq_emb_cal

    def get_classifier(self, category_names: List[str]):
        text_embed_bucket = []
        for template in self.templates:  # 对每一组模板，将类别名填入模板中，然后编码。
            noun_tokens = self.tokenizer(  # (类别数, 77)
                [template.format(noun) for noun in category_names]
            )
            text_inputs = noun_tokens.to(self.text_projection.data.device)
            text_embed = self.encode_text(text_inputs, normalize=True)
            text_embed_bucket.append(text_embed)
        text_embed = torch.stack(text_embed_bucket).mean(dim=0)  # 对每一组模板的编码取平均。
        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
        return text_embed

    @torch.no_grad()
    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)  # attn_mask是一个因果mask，用于避免attention到未来的token。
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return F.normalize(x, dim=-1) if normalize else x

    def get_classifier_by_vocabulary(self, vocabulary: List[str]):
        if self.cache_feature:
            new_words = [word for word in vocabulary if word not in self.cache]
            if len(new_words) > 0:
                cat_embeddings = self.get_classifier(new_words)
                self.cache.update(dict(zip(new_words, cat_embeddings)))
            cat_embeddings = torch.stack([self.cache[word] for word in vocabulary])
        else:
            cat_embeddings = self.get_classifier(vocabulary)
        return cat_embeddings

    def get_classifier_by_dataset_name(self, dataset_name: str):
        if self.cache_feature:
            if dataset_name not in self.cache:
                category_names = get_labelset_from_dataset(dataset_name)  # 从数据集名获得类别名。
                cat_embeddings = self.get_classifier(category_names)
                self.cache[dataset_name] = cat_embeddings  # 将数据集的类别向量缓存起来。
            cat_embeddings = self.cache[dataset_name]
        else:
            category_names = get_labelset_from_dataset(dataset_name)
            cat_embeddings = self.get_classifier(category_names)
        return cat_embeddings

    def _freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        super().train(False)

    def forward(self, feat: torch.Tensor, vocabulary: list[str]=None, dataset_name: str=None):
        assert vocabulary is None or dataset_name is None
        if vocabulary is not None:
            text_feat = self.get_classifier_by_vocabulary(vocabulary)
        elif dataset_name is not None:
            text_feat = self.get_classifier_by_dataset_name(dataset_name)
        elif self.vocabulary is not None:
            text_feat = self.get_classifier_by_vocabulary(self.vocabulary)
        elif self.dataset_name is not None:
            text_feat = self.get_classifier_by_dataset_name(self.dataset_name)
        else:
            raise ValueError("No vocabulary or dataset_name provided.")

        feat = feat / feat.norm(dim=-1, keepdim=True)  # (...,D)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * feat @ text_feat.t()  # (..., K) logit。
        if hasattr(self, "logit_bias"):
            logits += self.logit_bias

        return logits


class LearnableBgOvClassifier(PredefinedOvClassifier):

    def __init__(
            self,
            clip_model: CLIP,
            tokenizer: SimpleTokenizer | HFTokenizer,
            cache_feature: bool = True,
            templates: List[str] = ("a photo of {}",),
    ):
        super().__init__(clip_model, tokenizer, cache_feature, templates)
        self.bg_embed = nn.Parameter(torch.randn(1, self.text_projection.shape[0]))  # 可学的背景Proxy。
        nn.init.normal_(
            self.bg_embed,
            std=self.bg_embed.shape[1] ** -0.5,
        )

    def get_classifier_by_vocabulary(self, vocabulary: List[str]):
        cat_embedding = super().get_classifier_by_vocabulary(vocabulary)
        cat_embedding = torch.cat([cat_embedding, self.bg_embed], dim=0)
        cat_embedding = F.normalize(cat_embedding, p=2, dim=-1)
        return cat_embedding

    def get_classifier_by_dataset_name(self, dataset_name: str):
        cat_embedding = super().get_classifier_by_dataset_name(dataset_name)
        cat_embedding = torch.cat([cat_embedding, self.bg_embed], dim=0)
        cat_embedding = F.normalize(cat_embedding, p=2, dim=-1)  # 确保（背景）向量的范数为1。
        return cat_embedding
