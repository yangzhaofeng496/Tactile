from dataloader.cache_loader import AdjacentPairBatchSampler


class DatasetStub:
    episode_positions = [[0, 1, 2, 3], [4, 5, 6]]


def test_sampler_keeps_adjacent_windows_in_the_same_batch():
    sampler = AdjacentPairBatchSampler(
        DatasetStub(),
        fraction=1.0,
        seed=11,
        batch_size=4,
    )

    batches = list(iter(sampler))
    assert batches
    for batch in batches:
        for left, right in zip(batch[::2], batch[1::2]):
            assert right == left + 1

