import json
import mmap
import os
import struct
from contextlib import contextmanager
from typing import Dict, Any

import numpy as np
import torch
from torch import distributed as dist

def init_distributed():
    rank = int(os.getenv('RANK', '0'))
    world_size = int(os.getenv("WORLD_SIZE", '1'))

    backend = "gloo"

    # Call the init process.
    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '6000')
    init_method += master_ip + ':' + master_port
    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
        init_method=init_method
    )
    return True

class SafeTensorsWriter:
    def __init__(self, mm, headers, offset: int = 0):
        self.mm = mm
        self.headers = headers
        self.offset = offset

    def __getitem__(self, item):
        start, end = self.headers[item]["offsets"]
        return self.mm[start: end] # TODO: this actually read the bytes instead of assigning new ones

    def write_data(self, item, s: slice, data):
        start, end = self.headers[item]["offsets"]
        dtype = self.headers[item]["dtype"]
        # TODO @thomasw21: We need to support other slicing options
        assert s.start >= 0
        assert s.stop >= 0
        byte_per_elt = torch.finfo(dtype).bits // 8
        new_slice = slice(s.start * byte_per_elt + start + self.offset, s.stop * byte_per_elt + start + self.offset, s.step)
        self.mm[new_slice] = data

    def __delitem__(self, key):
        raise ValueError(f"Cannot destroy {key}")

    def __len__(self):
        return len(self.headers)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass



@contextmanager
def new_file(filename: str, headers: Dict[str, Dict[str, Any]]):
    current_offset = 0
    for name, header in headers.items():
        new_offset = current_offset + (torch.finfo(header["dtype"]).bits // 8) * np.prod(header["shape"])
        header["offsets"] = (current_offset, new_offset)
        current_offset = new_offset
    total_size_in_byte = current_offset

    # Compute headers size
    encoded_json_headers = json.dumps(
        headers,
        sort_keys=True,
        default=str # FIXME: quick and dirty fix to get serialization working
    ).encode("utf8")
    json_headers_length = len(encoded_json_headers)

    length = 8 + json_headers_length + total_size_in_byte
    with open(filename, mode="wb") as fi:
        # little endian, unsigned long
        fi.write(struct.pack("<Q", json_headers_length))
        fi.write(encoded_json_headers)
        # FIXME: current hack to preallocate file memory
        fi.seek(length-1)
        fi.write(bytes(True))
    with open(filename, mode="r+") as fi:
        offset = 8 + json_headers_length
        with mmap.mmap(fileno=fi.fileno(), length=length, offset=0, access=mmap.ACCESS_WRITE) as mm:
            with SafeTensorsWriter(mm=mm, headers=headers, offset=offset) as writer:
                yield writer


def main():
    hidden_size = 512
    filename = "my_weights.safetensors"
    init_distributed()

    tp_pg = dist.distributed_c10d._get_default_group()

    # my_weights are essentially `torch.arange(hidden_size)
    assert hidden_size % tp_pg.size() == 0
    chunk_size = hidden_size // tp_pg.size()

    print(chunk_size)
    offset = tp_pg.rank() * chunk_size
    my_weights = torch.arange(chunk_size, dtype=torch.float16) + offset

    headers = {
        "my_weight": {"dtype": my_weights.dtype, "shape": [hidden_size]}
    }
    with new_file(filename, headers=headers) as writer:
        # Assert `fo.get_tensor("my_weight")` would throw KeyError or something like that

        # Store data
        start = tp_pg.rank() * chunk_size
        end = (tp_pg.rank() + 1) * chunk_size

        # Annoying that this doesn't work because it seems writer["my_weight"] would return bytes instead of the setter ...
        # writer["my_weight"][start:end] = my_weights
        print(len(bytearray(my_weights.numpy())))
        writer.write_data("my_weight", slice(start, end), bytearray(my_weights.numpy()))


if __name__ == "__main__":
    main()