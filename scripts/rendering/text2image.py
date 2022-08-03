# DALLE MODEL
#DALLE_MODEL = "dalle-mini/dalle-mini/mega-1-fp16:latest"  # can be wandb artifact or 🤗 Hub or local folder or google bucket
DALLE_MODEL = "dalle-mini/dalle-mini/mini-1:v0"
DALLE_COMMIT_ID = None

# VQGAN model
VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"

import jax
import jax.numpy as jnp
from dalle_mini import DalleBart, DalleBartProcessor
from vqgan_jax.modeling_flax_vqgan import VQModel
from flax.jax_utils import replicate
from functools import partial
from flax.training.common_utils import shard_prng_key
import numpy as np
from PIL import Image
from tqdm.notebook import trange
import random

from scripts.util.logging import get_logger

LOG = get_logger(__name__)

from dalle_mini import DalleBart, DalleBartProcessor
from flax.jax_utils import replicate
from functools import partial
from flax.training.common_utils import shard_prng_key
import numpy as np
from PIL import Image
from tqdm.notebook import trange
import random


class Text2Image:

    def __init__(self, model_name: str, model_commit_id: str, key, model,
                 params, processor, vqgan_params, vqgan):
        self.model = model
        self.params = params
        self.processor = processor

        # number of predictions per prompt
        self.n_predictions = 8

        # We can customize generation parameters (see https://huggingface.co/blog/how-to-generate)
        self.gen_top_k = None
        self.gen_top_p = None
        self.temperature = None
        self.cond_scale = 10.0
        self.key = key
        self.vqgan_params = vqgan_params
        self.vqgan = vqgan

    def p_generate(self, tokenized_prompt):
        return _p_generate(self.model, tokenized_prompt, self.params,
                           self.gen_top_k, self.gen_top_p, self.temperature,
                           self.cond_scale)

    def p_decode(self, encoded_images):
        return _p_decode(self.vqgan, encoded_images, self.vqgan_params)

    def __call__(self, prompts):
        tokenized_prompts = self.processor(prompts)
        #tokenized_prompt = replicate(tokenized_promts)
        images = []
        for i in trange(max(self.n_predictions // jax.device_count(), 1)):
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
                display(img)
                print()


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
    # check how many devices are available
    jax.local_device_count()
    seed = random.randint(0, 2**32 - 1)
    key = jax.random.PRNGKey(seed)
    model, parser = DalleBart.from_pretrained(DALLE_MODEL,
                                              revision=DALLE_COMMIT_ID,
                                              dtype=jnp.float16,
                                              _do_init=False)
    processor = DalleBartProcessor.from_pretrained(DALLE_MODEL,
                                                   revision=DALLE_COMMIT_ID)
    # Load VQGAN
    vqgan, vqgan_params = VQModel.from_pretrained(VQGAN_REPO,
                                                  revision=VQGAN_COMMIT_ID,
                                                  _do_init=False)

    text_context = Text2Image(DALLE_MODEL, DALLE_COMMIT_ID, key, model, parser,
                              processor, vqgan_params, vqgan)

    prompts = [
        "sunset over a lake in the mountains",
        "the Eiffel tower landing on the moon",
    ]
    text_context(prompts)