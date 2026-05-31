import asr_bench


def test_vram_sampler_records_peak_of_injected_reads():
    reads = iter([100, 500, 300, 900, 200])

    def fake_read():
        try:
            return next(reads)
        except StopIteration:
            return 0

    s = asr_bench.VramSampler(read_fn=fake_read, interval=0.001)
    # Drive the recording logic deterministically rather than relying on thread timing.
    for _ in range(5):
        s._record(fake_read())
    assert s.peak == 900


def test_vram_sampler_peak_starts_at_zero():
    s = asr_bench.VramSampler(read_fn=lambda: 0, interval=0.01)
    assert s.peak == 0
