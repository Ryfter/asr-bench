import math
import struct
import wave
import asr_bench


def _write_sine_wav(path, seconds=1.0, rate=16000, freq=440.0):
    n = int(seconds * rate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        frames = b"".join(
            struct.pack("<h", int(0.3 * 32767 * math.sin(2 * math.pi * freq * i / rate)))
            for i in range(n)
        )
        wf.writeframes(frames)


def test_decode_to_pcm16_roundtrip(tmp_path):
    wav = tmp_path / "tone.wav"
    _write_sine_wav(wav, seconds=1.0, rate=16000)
    pcm, n_samples = asr_bench.decode_to_pcm16(wav)
    assert isinstance(pcm, (bytes, bytearray))
    assert len(pcm) == n_samples * 2          # s16le => 2 bytes/sample
    # ~16000 samples for 1 second at 16kHz (allow small resampler edge slack)
    assert 15500 <= n_samples <= 16500


def test_decode_to_pcm16_resamples_44k_to_16k(tmp_path):
    wav = tmp_path / "tone44.wav"
    _write_sine_wav(wav, seconds=1.0, rate=44100)
    pcm, n_samples = asr_bench.decode_to_pcm16(wav)
    assert 15500 <= n_samples <= 16500       # resampled down to 16kHz
