"""
Basic script that downloads and caches a huggingface dataset to disk
"""
import sys
import logging

from datasets import load_dataset
from scripts.util.logging import get_logger

LOG = logging.getLogger(__name__)


def main():
    dataset_name = sys.argv[1]
    dataset_split = sys.argv[2]
    cache_dir = sys.argv[3]
    auth_token = sys.argv[4]

    LOG.info("Start loading data")
    load_dataset(
        dataset_name,
        split=dataset_split,
        use_auth_token=auth_token,
        cache_dir=cache_dir,
    )
    LOG.info("Finished loading data")


if __name__ == "__main__":
    main()
