from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader, BatchSampler, RandomSampler, Sampler


class HFSampler(Sampler):
    def __init__(self, ds: Dataset, batch_size: int, drop_last: bool):
        super().__init__(ds)
        self.ds = ds
        self.batch_size = batch_size

    def __iter__(self):
        for end in range(self.batch_size, len(self), self.batch_size):
            start = end - self.batch_size
            yield self.ds[start:end]

    def __len__(self):
        return len(self.ds)


def main():
    ds = load_dataset("wikitext", "wikitext-2-v1", split="test")[:5]["text"]

    # loader = DataLoader(
    #     dataset=ds,
    #     num_workers=1,
    #     sampler=HFSampler(ds, batch_size=2, drop_last=True)
    # )
    loader = DataLoader(
        dataset=ds,
        batch_sampler=BatchSampler(RandomSampler(ds), batch_size=2, drop_last=True)
    )

    for batch in loader:
        print(batch)


if __name__ == "__main__":
    main()