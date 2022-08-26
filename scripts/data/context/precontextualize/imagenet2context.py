import argparse
from datasets import load_dataset


def main(args: argparse.Namespace):
    image_net = load_dataset("imagenet-1k",
                             use_auth_token=args.access_token,
                             split="train",
                             streaming=True)
    dataset_stats = {
        "total_uploaded_size": 0,
        "total_dataset_nbytes": 0,
        "total_num_shards": 0,
        "total_num_examples": 0,
        "total_num_words": 0,
    }

    for image_id, (image_dict) in enumerate(image_net):
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        help="Path to Text2Image json file",
    )
    parser.add_argument(
        "--access_token",
        type=str,
        help="Huggingface access token with for the download of imagenet",
    )
    parsed_args = parser.parse_args()
    main(parsed_args)