import numpy as np
import random

from typing import Dict, Any, List
from dataclasses import dataclass
from functools import partial
from PIL import Image
from tqdm import trange

from dalle_mini import DalleBart, DalleBartProcessor
from vqgan_jax.modeling_flax_vqgan import VQModel

import jax
import jax.numpy as jnp
from flax.jax_utils import replicate
from flax.training.common_utils import shard_prng_key


# Text2Image Config class
@dataclass
class Text2ImageConfig:
    generator_version: str
    generator_commit_id: str
    decoder_version: str
    decoder_commit_id: str
    n_predictions: int
    # We can customize generation parameters
    # (see https://huggingface.co/blog/how-to-generate)
    gen_top_k: int = None
    gen_top_p: int = None
    temperature: int = None
    cond_scale: int = 10.0
    num_sentences: int = 2

    @classmethod
    def from_dict(cls, text2image_dict: Dict[str, Any]):
        return cls(**text2image_dict)


class Text2Image:

    def __init__(self, config):
        self.config = config
        self.model, params = DalleBart.from_pretrained(
            config.generator_version,
            revision=config.generator_commit_id,
            dtype=jnp.float16,
            _do_init=False)
        self.params = replicate(params)
        self.processor = DalleBartProcessor.from_pretrained(
            config.generator_version, revision=config.generator_commit_id)
        self.decoder, decoder_params = VQModel.from_pretrained(
            config.decoder_version,
            revision=config.decoder_commit_id,
            _do_init=False)
        self.decoder_params = replicate(decoder_params)

    def create_prompts(self, text):
        text = text.split("\n")
        if self.config.num_sentences:
            prompts = []
            idx = 0
            for paragraph in text:
                if paragraph:
                    paragraph = paragraph.split(".")
                    sentence_group = ""
                    for sentence in paragraph:
                        if sentence:
                            sentence_group += sentence
                            idx += 1
                            if idx % self.config.num_sentences == 0:
                                prompts.append(sentence_group)
                                sentence_group = ""
                    if sentence_group:
                        prompts.append(sentence_group)
                        sentence_group = ""
        return prompts


# model inference
@partial(jax.pmap,
         axis_name="batch",
         static_broadcasted_argnums=(3, 4, 5, 6, 7))
def p_generate(tokenized_prompt, key, params, top_k, top_p, temperature,
               condition_scale, model):
    return model.generate(
        **tokenized_prompt,
        prng_key=key,
        params=params,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        condition_scale=condition_scale,
    )


# decode image
@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(2))
def p_decode(indices, params, model):
    return model.decode_code(indices, params=params)


def generate_from_prompts(config, tokenized_prompt, generator,
                          generator_params, decoder, decoder_params, key):
    images = []
    for i in trange(max(config.n_predictions // jax.device_count(), 1)):
        key, subkey = jax.random.split(key)
        # generate images
        encoded_images = p_generate(tokenized_prompt, shard_prng_key(subkey),
                                    generator_params, config.gen_top_k,
                                    config.gen_top_p, config.temperature,
                                    config.cond_scale, generator)

        encoded_images = encoded_images.sequences[..., 1:]
        # decode images
        decoded_images = p_decode(encoded_images, decoder_params, decoder)
        decoded_images = decoded_images.clip(0.0, 1.0).reshape(
            (-1, 256, 256, 3))
        for decoded_img in decoded_images:
            img = Image.fromarray(np.asarray(decoded_img * 255,
                                             dtype=np.uint8))
            images.append(img)
    return images


def text_to_sentence(text):
    text = text.split("\n")
    sentences = []
    for paragraph in text:
        paragraph = paragraph.strip()
        if paragraph:
            paragraph = paragraph.split(".")
            for sentence in paragraph:
                sentence = sentence.strip()
                if sentence:
                    sentences.append(sentence)
    return sentences


def sentence_to_prompts(sentences: List, num_sentences: int):
    prompts = []
    idx = 1
    prompt = ''
    for sentence in sentences:
        prompt += sentence
        if idx % num_sentences == 0:
            prompts.append(prompt)
            prompt = ''
        idx += 1
    if prompt:
        prompts.append(prompt)
    return prompts