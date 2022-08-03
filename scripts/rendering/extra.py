# DALLE MODEL
#DALLE_MODEL = "dalle-mini/dalle-mini/mega-1-fp16:latest"  # can be wandb artifact or 🤗 Hub or local folder or google bucket
DALLE_MODEL = "dalle-mini/dalle-mini/mini-1:v0"
DALLE_COMMIT_ID = None

# VQGAN model
VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"

import jax
import jax.numpy as jnp
import numpy as np
import random
from dataclasses import dataclass
from flax.jax_utils import replicate
from functools import partial
from PIL import Image
from tqdm import trange
from tqdm.notebook import trange
from typing import Dict, Any

from dalle_mini import DalleBart, DalleBartProcessor
from vqgan_jax.modeling_flax_vqgan import VQModel

from scripts.util.logging import get_logger

LOG = get_logger(__name__)


@dataclass
class Text2ImageConfig:
    generator_version: str
    generator_commit_id: str
    decoder_version: str
    decoder_commit_id: str
    n_predictions: int
    gen_top_k: int = None
    gen_top_p: int = None
    temperature: int = None
    cond_scale: int = 10.0

    @classmethod
    def from_dict(cls, text2image_dict: Dict[str, Any]) -> Text2ImageConfig:
        return cls(**text2image_dict)


class Text2Image:

    def __init__(self, generator_version: str, generator_commit_id: str,
                 decoder_version: str, decoder_commit_id: str,
                 n_predictions: int):
        self.generator_model, self.generator_params = DalleBart.from_pretrained(
            generator_version,
            revision=generator_commit_id,
            dtype=jnp.float16,
            _do_init=False,
        )
        self.generator_processor = DalleBartProcessor.from_pretrained(
            generator_version, revision=generator_commit_id)
        self.decoder_model, self.decoder_params = VQModel.from_pretrained(
            decoder_version, revision=decoder_commit_id, _do_init=False)

        self.n_predictions = n_predictions

        LOG.info(
            f"N Devices used for the inference: {jax.local_device_count()}")

        # We can customize generation parameters (see https://huggingface.co/blog/how-to-generate)
        self.gen_top_k = None
        self.gen_top_p = None
        self.temperature = None
        self.cond_scale = 10.0
        #seed = random.randint(0, 2**32 - 1)
        #self.key = jax.random.PRNGKey(seed)

    def prompt_generate(self, tokenized_prompt):
        return _p_generate(self.generator_model, tokenized_prompt,
                           self.generator_params, self.gen_top_k,
                           self.gen_top_p, self.temperature, self.cond_scale)

    def prompt_decode(self, encoded_images):
        return _p_decode(self.decoder_model, encoded_images,
                         self.decoder_params)

    def __call__(self, prompts):
        tokenized_prompts = self.generator_processor(prompts)
        #tokenized_prompt = replicate(tokenized_promts)
        images = []
        for i in trange(max(self.n_predictions // jax.device_count(), 1)):
            #self.key, subkey = jax.random.split(self.key)
            encoded_images = self.prompt_generate(
                tokenized_prompts)  #,shard_prng_key(subkey))
            encoded_images = encoded_images.sequences[..., 1:]
            decoded_images = self.prompt_decode(encoded_images)
            decoded_images = decoded_images.clip(0.0, 1.0).reshape(
                (-1, 256, 256, 3))
            for decoded_img in decoded_images:
                img = Image.fromarray(
                    np.asarray(decoded_img * 255, dtype=np.uint8))
                images.append(img)
                display(img)
                print()


# TODO: add JAX support
# Model Inference
#@partial(jax.pmap, axis_name="batch")#, static_broadcasted_argnums=(4, 5, 6, 7))
def _p_generate(model, tokenized_prompt, params, top_k, top_p, temperature,
                condition_scale):
    return model.generate(
        **tokenized_prompt,
        #prng_key=key,
        params=params,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        condition_scale=condition_scale,
    )


# decode image
#@partial(jax.pmap, axis_name="batch")
def _p_decode(vqgan, indices, params):
    return vqgan.decode_code(indices, params=params)


if '__main__' == __name__:
    image_context_generator = Text2Image(DALLE_MODEL, DALLE_COMMIT_ID,
                                         VQGAN_REPO, VQGAN_COMMIT_ID, 1)
    prompts = ["sunset over a lake in the mountains"]
    image_context_generator(prompts)