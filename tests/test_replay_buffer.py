"""
Tests for the replay buffer's store of deaths (ReplayBuffer with death_share > 0).

Each test adds transitions numbered 0, 1, 2, ... (the number is stored as the reward); every 10th one is a death.

Run from the repo root: python -m pytest tests/test_replay_buffer.py
"""

import torch as t

from dqn.replay_buffer import ReplayBuffer

NUM_ENVS = 4
SIZE, DEATH_SHARE = 100, 0.1  # the death store holds the latest 10 deaths


def make_buffer(death_share: float = DEATH_SHARE) -> ReplayBuffer:
    return ReplayBuffer(NUM_ENVS, (1,), (), SIZE, seed=0, device="cpu", death_share=death_share)


def fill(buffer: ReplayBuffer, num_transitions: int, keep: t.Tensor | None = None) -> None:
    """Adds transitions 0..num_transitions-1; transition n is a death when n is a multiple of 10."""
    for start in range(0, num_transitions, NUM_ENVS):
        numbers = t.arange(start, start + NUM_ENVS, dtype=t.float32)
        deaths = numbers % 10 == 0
        buffer.add(numbers[:, None], t.zeros(NUM_ENVS, dtype=t.int64), numbers, deaths, numbers[:, None], keep)


def test_without_death_share_it_is_a_plain_circular_buffer():
    buffer = make_buffer(death_share=0.0)
    fill(buffer, 400)
    assert buffer.size == SIZE and buffer.deaths.capacity == 0
    assert buffer.rewards.tolist() == list(range(300, 400))  # the latest 100, oldest first


def test_death_store_keeps_the_latest_deaths():
    buffer = make_buffer()
    fill(buffer, 400)
    assert sorted(buffer.deaths.in_order("rewards").tolist()) == list(range(300, 400, 10))
    assert buffer.deaths.in_order("terminated").all()


def test_deaths_are_kept_after_they_leave_the_main_store():
    buffer = make_buffer()
    fill(buffer, 20)  # deaths 0 and 10
    for start in range(1000, 1400, NUM_ENVS):  # 400 more transitions, none of them deaths
        numbers = t.arange(start, start + NUM_ENVS, dtype=t.float32)
        buffer.add(numbers[:, None], t.zeros(NUM_ENVS, dtype=t.int64), numbers, t.zeros(NUM_ENVS, dtype=t.bool),
                   numbers[:, None])
    assert not buffer.terminated.any()  # the main store has no deaths left
    assert sorted(buffer.deaths.in_order("rewards").tolist()) == [0.0, 10.0]


def test_batches_take_the_death_share_from_the_death_store():
    buffer = make_buffer()
    fill(buffer, 400)
    batch = buffer.sample(1000)
    assert batch.terminated.sum().item() >= 100  # 10% from the death store, plus any deaths drawn from the main store


def test_without_any_deaths_yet_every_sample_comes_from_the_main_store():
    buffer = make_buffer()
    keep = t.tensor([False, True, True, True])  # drops transition 0, the only death among 0..3
    fill(buffer, NUM_ENVS, keep)
    assert buffer.deaths.count == 0
    assert set(buffer.sample(100).rewards.tolist()) <= {1.0, 2.0, 3.0}
