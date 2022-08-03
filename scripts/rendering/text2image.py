# DALLE MODEL
#DALLE_MODEL = "dalle-mini/dalle-mini/mega-1-fp16:latest"  # can be wandb artifact or 🤗 Hub or local folder or google bucket
DALLE_MODEL = "dalle-mini/dalle-mini/mini-1:v0"
DALLE_COMMIT_ID = None

# VQGAN model
VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"

import json
from dataclasses import dataclass
import numpy as np
from PIL import Image
from tqdm import trange
from typing import Dict, Any
import random
import jax
import jax.numpy as jnp
from dalle_mini import DalleBart, DalleBartProcessor
from vqgan_jax.modeling_flax_vqgan import VQModel
from flax.jax_utils import replicate
from functools import partial
from flax.training.common_utils import shard_prng_key

from scripts.util.logging import get_logger

LOG = get_logger(__name__)

# Enabling TPUs in GColab
try:
    import jax.tools.colab_tpu
    jax.tools.colab_tpu.setup_tpu()
except Exception as e:
    LOG.info("There is no TPU available")


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

    @classmethod
    def from_dict(cls, text2image_dict: Dict[str, Any]):
        return cls(**text2image_dict)


class Text2Image:

    def __init__(self, generator, params, processor, decoder, decoder_params,
                 config):
        self.generator = generator
        self.params = params
        self.processor = processor
        self.decoder = decoder
        self.decoder_params = decoder_params
        self.config = config
        #self.key = key

    def p_generate(self, tokenized_prompt):
        return _p_generate(self.generator, tokenized_prompt, self.params,
                           self.config.gen_top_k, self.config.gen_top_p,
                           self.config.temperature, self.config.cond_scale)

    def p_decode(self, encoded_images):
        return _p_decode(self.decoder, encoded_images, self.decoder_params)

    def __call__(self, prompts):
        tokenized_prompts = self.processor(prompts)
        #tokenized_prompt = replicate(tokenized_promts)
        images = []
        for i in trange(max(self.config.n_predictions // jax.device_count(),
                            1)):
            #self.key, subkey = jax.random.split(self.key)
            encoded_images = self.p_generate(tokenized_prompts)  #,
            #shard_prng_key(subkey))
            encoded_images = encoded_images.sequences[..., 1:]
            decoded_images = self.p_decode(encoded_images)
            decoded_images = decoded_images.clip(0.0, 1.0).reshape(
                (-1, 256, 256, 3))
            for decoded_img in decoded_images:
                img = Image.fromarray(
                    np.asarray(decoded_img * 255, dtype=np.uint8))
                images.append(img)
        return images


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
def _p_decode(model, indices, params):
    return model.decode_code(indices, params=params)


if '__main__' == __name__:
    # check how many devices are available
    LOG.info(f"N of devices available {jax.local_device_count()}")
    config_path = "./config/rendering/text2image_rendering.json"
    config = Text2ImageConfig.from_dict(json.load(open(config_path)))
    #seed = random.randint(0, 2**32 - 1)
    #key = jax.random.PRNGKey(seed)
    model, parser = DalleBart.from_pretrained(
        config.generator_version,
        revision=config.generator_commit_id,
        dtype=jnp.float16,
        _do_init=False)

    processor = DalleBartProcessor.from_pretrained(
        config.generator_version,
        revision=config.generator_commit_id,
    )
    # Load VQGAN
    decoder, decoder_params = VQModel.from_pretrained(
        config.decoder_version,
        revision=config.decoder_commit_id,
        _do_init=False)

    text_context = Text2Image(model, parser, processor, decoder,
                              decoder_params, config)

    prompts = [
        "sunset over a lake in the mountains",
        "the Eiffel tower landing on the moon",
    ]
    text_context(prompts)