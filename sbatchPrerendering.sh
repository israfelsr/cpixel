export PYTHONPATH=$(pwd)
export TOKENIZERS_PARALLELISM=false
python3 scripts/data/context/precontextualize/bookcorpus2context.py\
    --config_file="./config/rendering/text2image_rendering.json"
