import threading
import time

import pytest

from padel_engine import video
from padel_engine.video import batched, h264_args, prefetch


def test_prefetch_keeps_order_and_everything():
    assert list(prefetch(iter(range(100)), 4)) == list(range(100))
    assert list(prefetch(iter(range(5)), 0)) == list(range(5))


def test_prefetch_stops_its_thread_when_the_consumer_stops():
    closed = threading.Event()

    def frames():
        try:
            for i in range(10_000):
                yield i
        finally:
            closed.set()

    it = prefetch(frames(), 2)
    assert next(it) == 0
    it.close()
    assert closed.wait(5)
    time.sleep(0.05)
    assert not any(t.name == "prefetch" and t.is_alive() for t in threading.enumerate())


def test_prefetch_raises_the_producers_error():
    def frames():
        yield 1
        raise ValueError("bad frame")

    it = prefetch(frames(), 2)
    assert next(it) == 1
    with pytest.raises(ValueError, match="bad frame"):
        next(it)


def test_batched():
    assert list(batched(range(7), 3)) == [[0, 1, 2], [3, 4, 5], [6]]


def test_encoder_choice(monkeypatch):
    assert h264_args(23, "libx264")[:2] == ["-c:v", "libx264"]
    nvenc = h264_args(20, "h264_nvenc")
    assert nvenc[:2] == ["-c:v", "h264_nvenc"] and nvenc[nvenc.index("-cq") + 1] == "20"
    monkeypatch.setattr(video, "nvenc_available", lambda: False)
    assert h264_args(23)[:2] == ["-c:v", "libx264"]
    monkeypatch.setattr(video, "nvenc_available", lambda: True)
    assert h264_args(23)[:2] == ["-c:v", "h264_nvenc"]
