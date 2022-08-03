#DALLE_MODEL = "dalle-mini/dalle-mini/mega-1-fp16:latest"  # can be wandb artifact or 🤗 Hub or local folder or google bucket
DALLE_COMMIT_ID = None

# if the notebook crashes too often you can use dalle-mini instead by uncommenting below line
DALLE_MODEL = "dalle-mini/dalle-mini/mini-1:v0" ""

from dalle_mini import DalleBart, DalleBartProcessor
import jax
import jax.numpy as jnp
from flax.jax_utils import replicate
from functools import partial

from scripts.util.logging import get_logger
import random

LOG = get_logger(__name__)


class Text2Image:

    def __init__(self, model_version: str, model_commit_id: str,
                 num_predictions: int):
        self.model, params, self.processor = self.load_model(
            model_version, model_commit_id)
        self.params = replicate(params)
        self.num_predictions = num_predictions
        LOG.info(
            f"N Devices used for the inference: {jax.local_device_count()}")

        # We can customize generation parameters (see https://huggingface.co/blog/how-to-generate)
        self.gen_top_k = None
        self.gen_top_p = None
        self.temperature = None
        self.cond_scale = 10.0
        seed = random.randint(0, 2**32 - 1)
        self.key = jax.random.PRNGKey(seed)

    def load_model(self, model_version, model_commit_id):
        model, params = DalleBart.from_pretrained(model_version,
                                                  revision=model_commit_id,
                                                  dtype=jnp.float16,
                                                  _do_init=False)
        processor = DalleBartProcessor.from_pretrained(
            model_version, revision=model_commit_id)
        return model, params, processor

    @partial(jax.pmap,
             axis_name="batch",
             static_broadcasted_argnums=(3, 4, 5, 6))
    def p_generate(self, tokenized_prompt, key):
        return self.model.generate(**tokenized_prompt,
                                   prng_key=key,
                                   params=self.params,
                                   top_k=self.top_k,
                                   top_p=self.top_p,
                                   temperature=self.temperature,
                                   condition_scale=self.condition_scale)

    def __call__(self, prompts: str):
        LOG.info(f"Generating images for {prompts}")
        tokenized_prompts = self.text_procesor(prompts)
        tokenized_prompt = replicate(tokenized_prompts)
        key, subkey = jax.random.split(self.key)
        encoded_images = self.p_generate(tokenized_prompt, key)
        print(encoded_images)
        encoded_images = encoded_images.sequences[..., 1:]
        print(encoded_images)


if '__main__' == __name__:
    image_context_generator = Text2Image(DALLE_MODEL, DALLE_COMMIT_ID, 2)
    prompts = ["sunset over a lake in the mountains"]
    image_context_generator(prompts)
