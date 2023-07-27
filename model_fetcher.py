'''
RunPod | Transformer | Model Fetcher
'''

import argparse

import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM)


def download_model(model_name):
    AutoModelForCausalLM.from_pretrained(model_name,
                                         trust_remote_code=True)
    AutoTokenizer.from_pretrained(model_name)


# ---------------------------------------------------------------------------- #
#                                Parse Arguments                               #
# ---------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model_name", type=str,
                    default="mosaicml/mpt-30b-instruct", help="HuggingFace URI of the model to download.")


if __name__ == "__main__":
    args = parser.parse_args()
    download_model(args.model_name)
