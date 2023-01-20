import argparse
import json
import os
from functools import partial
from multiprocessing import Pool

import torch

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download, CommitInfo
from safetensors.torch import save_file


def rename(pt_filename) -> str:
    local = pt_filename.replace(".bin", ".safetensors")
    local = local.replace("pytorch_model", "model")
    return local

def convert_multi(api: HfApi, model_id, num_proc: int = 1) -> CommitInfo:
    local_filenames = []
    try:
        filename = hf_hub_download(
            repo_id=model_id, filename="pytorch_model.bin.index.json"
        )
        with open(filename, "r") as f:
            data = json.load(f)

        filenames = set(data["weight_map"].values())
        num_proc = min(num_proc, len(filenames))

        shard_filenames = []
        if num_proc > 1:
            with Pool(num_proc) as pool:
                cached_filenames = pool.imap(partial(hf_hub_download, model_id), filenames)
                for cached_filename, filename in zip(cached_filenames, filenames):
                    loaded = torch.load(cached_filename)
                    local = rename(filename)
                    print(local)
                    save_file(loaded, local, metadata={"format": "pt"})
                    shard_filenames.append(local)
        else:
            for filename in filenames:
                cached_filename = hf_hub_download(repo_id=model_id, filename=filename)
                loaded = torch.load(cached_filename)
                local = rename(filename)
                save_file(loaded, local, metadata={"format": "pt"})
                shard_filenames.append(local)
        local_filenames = local_filenames + shard_filenames

        print("Writing index file")
        index = "model.safetensors.index.json"
        with open(index, "w") as f:
            newdata = {k: v for k, v in data.items()}
            newmap = {k: rename(v) for k, v in data["weight_map"].items()}
            newdata["weight_map"] = newmap
            json.dump(newdata, f)
        local_filenames.append(index)

        print("Pushing and creating PR")
        operations = [
            CommitOperationAdd(path_in_repo=local, path_or_fileobj=local)
            for local in local_filenames
        ]
        return api.create_commit(
            repo_id=model_id,
            operations=operations,
            commit_message="Adding `safetensors` variant of this model",
            commit_description="Converted from this Space: https://huggingface.co/spaces/safetensors/convert",
            create_pr=True,
        )
    finally:
        for local in local_filenames:
            pass
            # os.remove(local)


def convert_single(api: HfApi, model_id) -> CommitInfo:
    local = "model.safetensors"
    try:
        filename = hf_hub_download(repo_id=model_id, filename="pytorch_model.bin")
        loaded = torch.load(filename)
        save_file(loaded, local, metadata={"format": "pt"})

        operations = [CommitOperationAdd(path_in_repo=local, path_or_fileobj=local)]
        return api.create_commit(
            repo_id=model_id,
            operations=operations,
            commit_message="Adding `safetensors` variant of this model",
            commit_description="Converted from this Space: https://huggingface.co/spaces/safetensors/convert",
            create_pr=True,
        )
    finally:
        os.remove(local)


def convert(token: str, model_id: str) -> CommitInfo:
    """
    returns url to the PR
    """
    api = HfApi(token=token)
    info = api.model_info(model_id)
    filenames = set(s.rfilename for s in info.siblings)
    if "pytorch_model.bin" in filenames:
        return convert_single(api, model_id)
    elif "pytorch_model.bin.index.json" in filenames:
        return convert_multi(api, model_id)
    raise ValueError("repo does not seem to have a pytorch_model in it")


if __name__ == "__main__":
    DESCRIPTION = """
    Simple utility tool to convert automatically some weights on the hub to `safetensors` format.
    It is PyTorch exclusive for now.
    It works by downloading the weights (PT), converting them locally, and uploading them back
    as a PR on the hub.
    """
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "model_id",
        type=str,
        help="The name of the model on the hub to convert. E.g. `gpt2` or `facebook/wav2vec2-base-960h`",
    )
    parser.add_argument(
        "--num-proc",
        type=int
    )
    args = parser.parse_args()
    model_id = args.model_id
    api = HfApi()
    info = api.model_info(model_id)
    filenames = set(s.rfilename for s in info.siblings)
    if "pytorch_model.bin" in filenames:
        convert_single(api, model_id)
    else:
        convert_multi(api, model_id, args.num_proc)