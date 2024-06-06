# COPY: SAN/san/model/clip_utils/utils.py
import os
import os.path as osp
from dataclasses import dataclass
from typing import List

import open_clip
from open_clip import CLIP
from open_clip.tokenizer import HFTokenizer, SimpleTokenizer
from torchvision.transforms import Compose

PREDEFINED_LABELSETS = {}

PREDEFINED_TEMPLATES = {
    "imagenet": [
        "a bad photo of a {}.",
        "a photo of many {}.",
        "a sculpture of a {}.",
        "a photo of the hard to see {}.",
        "a low resolution photo of the {}.",
        "a rendering of a {}.",
        "graffiti of a {}.",
        "a bad photo of the {}.",
        "a cropped photo of the {}.",
        "a tattoo of a {}.",
        "the embroidered {}.",
        "a photo of a hard to see {}.",
        "a bright photo of a {}.",
        "a photo of a clean {}.",
        "a photo of a dirty {}.",
        "a dark photo of the {}.",
        "a drawing of a {}.",
        "a photo of my {}.",
        "the plastic {}.",
        "a photo of the cool {}.",
        "a close-up photo of a {}.",
        "a black and white photo of the {}.",
        "a painting of the {}.",
        "a painting of a {}.",
        "a pixelated photo of the {}.",
        "a sculpture of the {}.",
        "a bright photo of the {}.",
        "a cropped photo of a {}.",
        "a plastic {}.",
        "a photo of the dirty {}.",
        "a jpeg corrupted photo of a {}.",
        "a blurry photo of the {}.",
        "a photo of the {}.",
        "a good photo of the {}.",
        "a rendering of the {}.",
        "a {} in a video game.",
        "a photo of one {}.",
        "a doodle of a {}.",
        "a close-up photo of the {}.",
        "a photo of a {}.",
        "the origami {}.",
        "the {} in a video game.",
        "a sketch of a {}.",
        "a doodle of the {}.",
        "a origami {}.",
        "a low resolution photo of a {}.",
        "the toy {}.",
        "a rendition of the {}.",
        "a photo of the clean {}.",
        "a photo of a large {}.",
        "a rendition of a {}.",
        "a photo of a nice {}.",
        "a photo of a weird {}.",
        "a blurry photo of a {}.",
        "a cartoon {}.",
        "art of a {}.",
        "a sketch of the {}.",
        "a embroidered {}.",
        "a pixelated photo of a {}.",
        "itap of the {}.",
        "a jpeg corrupted photo of the {}.",
        "a good photo of a {}.",
        "a plushie {}.",
        "a photo of the nice {}.",
        "a photo of the small {}.",
        "a photo of the weird {}.",
        "the cartoon {}.",
        "art of the {}.",
        "a drawing of the {}.",
        "a photo of the large {}.",
        "a black and white photo of a {}.",
        "the plushie {}.",
        "a dark photo of a {}.",
        "itap of a {}.",
        "graffiti of the {}.",
        "a toy {}.",
        "itap of my {}.",
        "a photo of a cool {}.",
        "a photo of a small {}.",
        "a tattoo of the {}.",
    ],
    "vild": [
        "a photo of a {}.",
        "This is a photo of a {}",
        "There is a {} in the scene",
        "There is the {} in the scene",
        "a photo of a {} in the scene",
        "a photo of a small {}.",
        "a photo of a medium {}.",
        "a photo of a large {}.",
        "This is a photo of a small {}.",
        "This is a photo of a medium {}.",
        "This is a photo of a large {}.",
        "There is a small {} in the scene.",
        "There is a medium {} in the scene.",
        "There is a large {} in the scene.",
    ],
}


def get_labelset_from_dataset(dataset_name: str) -> List[str]:
    labelset = PREDEFINED_LABELSETS[dataset_name]
    return labelset


def get_predefined_templates(template_set_name: str) -> List[str]:
    if template_set_name not in PREDEFINED_TEMPLATES:
        raise ValueError(f"Template set {template_set_name} not found")
    return PREDEFINED_TEMPLATES[template_set_name]


@dataclass
class OpenCLIPCreated(object):
    """Output of OpenCLIPredictor"""
    model: CLIP = None
    preprocess_train: Compose = None
    preprocess_val: Compose = None
    tokenizer: HFTokenizer | SimpleTokenizer = None

    @classmethod
    def from_open_clip_create(cls, open_clip_create_ret: tuple[CLIP, Compose, Compose]):
        return cls(*open_clip_create_ret)

    @classmethod
    def create(cls, model_name: str, **create_kwargs):
        model, preprocess_train, preprocess_val = \
            open_clip.create_model_and_transforms(model_name,
                                                  cache_dir=osp.join(os.getcwd(), 'pretrains', 'open_clip'),
                                                  **create_kwargs)
        tokenizer = open_clip.get_tokenizer(model_name)
        return cls(model=model, preprocess_train=preprocess_train, preprocess_val=preprocess_val, tokenizer=tokenizer)
