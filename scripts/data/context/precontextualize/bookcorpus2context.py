"""
Script used to generate images from the bookcorpus from https://huggingface.co/datasets/bookcorpusopen
Processes the dataset book-by-book and uploads the generated examples in chunks to HuggingFace.
Examples are stored and compressed in parquet files.
Relies on a modified version of the datasets library installed through git submodule.
"""

import argparse
import json
#import logging

from datasets import load_dataset
from pixel.src.pixel.utils.prerendering import push_rendered_chunk_to_hub

from scripts.util.logging import get_logger
from scripts.data.context.text2image import *

LOG = get_logger(__name__)


def main(args: argparse.Namespace):
    # Load text2image from config
    config = Text2ImageConfig.from_dict(json.load(open(args.config_file)))
    text_to_image = Text2Image(config)
    key = jax.random.PRNGKey(args.seed)

    data = {"pixel_values": [], "text": []}
    dataset_stats = {
        "total_uploaded_size": 0,
        "total_dataset_nbytes": 0,
        "total_num_shards": 0,
        "total_num_examples": 0,
        "total_num_words": 0,
    }

    bookcorpus = load_dataset("bookcorpusopen", split="train", streaming=True)
    if args.num_senteces:
        config.num_sentences = args.num_sentences
    LOG.info(config)
    idx = 0

    for book_id, book in enumerate(bookcorpus):
        num_examples = idx

        LOG.info(
            f"{book_id}: {book['title']}, {config.num_sentences=}, {num_examples=}"
        )

        book_sentences = text_to_sentence(book["text"])
        book_prompts = sentence_to_prompts(book_sentences,
                                           config.num_sentences)
        num_images += len(book_prompts)
        prompts_batch = []
        for i in range(0, len(book_prompts), args.batch_size):
            prompts_batch.append(book_prompts[i:i + args.batch_size])
        for batch in prompts_batch:
            idx += len(batch)
            tokenized_prompts = text_to_image.processor(batch)
            tokenized_prompt = replicate(
                tokenized_prompts
            )  #this should send the other batch to the TPUs
            images = generate_from_prompts(text_to_image.config,
                                           tokenized_prompt,
                                           text_to_image.params,
                                           text_to_image.decoder,
                                           text_to_image.decoder_params, key)
            data["pixel_values"].extend(images)
            data["text"].extend(batch)
            if len(data["pixel_values"]) > args.chunk_size:
                # log example?
                dataset_stats = push_rendered_chunk_to_hub(
                    args, data, dataset_stats, idx)
                data = {"pixel_values": [], "text": []}
    if len(data["pixel_values"]) > 0:
        dataset_stats = push_rendered_chunk_to_hub(args, data, dataset_stats,
                                                   idx)
        data = {"pixel_values": [], "text": []}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        help="Path to Text2Image json file",
    )
    parser.add_argument(
        "--num_sentence",
        type=int,
        help="Number of senteces per image generation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch of prompts to generate",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100000,
        help="Push data to hub in chunks of N lines",
    )
    parser.add_argument("--repo_id",
                        type=str,
                        help="Name of dataset to upload")
    parser.add_argument("--split",
                        type=str,
                        help="Name of dataset split to upload")
    parser.add_argument(
        "--auth_token",
        type=str,
        help="Huggingface auth token with write access to the repo id",
    )

    parsed_args = parser.parse_args()
    LOG.info(parsed_args)
    main(parsed_args)
