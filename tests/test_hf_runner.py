import hf_runner


def test_arg_parser_required_and_defaults():
    ns = hf_runner.build_arg_parser().parse_args(
        ["--audio", "a.wav", "--model", "facebook/wav2vec2-large-960h"])
    assert ns.audio == "a.wav"
    assert ns.model == "facebook/wav2vec2-large-960h"
    assert ns.device == "cuda"
    assert ns.language == "en"
    assert ns.chunk_len_secs == 30.0
    assert ns.stride_len_secs == 5.0


def test_needs_decode():
    assert hf_runner._needs_decode("clip.wav") is False
    assert hf_runner._needs_decode("CLIP.WAV") is False
    assert hf_runner._needs_decode("lecture.mp4") is True
    assert hf_runner._needs_decode("a.m4a") is True


def test_chunks_to_segments_and_words_from_pipeline_output():
    # transformers ASR pipeline with return_timestamps="word" yields:
    #   {"text": "...", "chunks": [{"text": "hello", "timestamp": (0.0, 0.5)}, ...]}
    pipe_out = {"text": "hello there",
                "chunks": [{"text": "hello", "timestamp": (0.0, 0.5)},
                           {"text": "there", "timestamp": (0.5, 1.0)}]}
    words = hf_runner._words_from_chunks(pipe_out["chunks"])
    assert words == [{"word": "hello", "start": 0.0, "end": 0.5},
                     {"word": "there", "start": 0.5, "end": 1.0}]


def test_words_handle_missing_or_none_timestamp():
    chunks = [{"text": "x", "timestamp": (None, None)}]
    words = hf_runner._words_from_chunks(chunks)
    assert words == [{"word": "x", "start": 0.0, "end": 0.0}]


def test_segments_group_words_into_cues():
    words = [{"word": "hello", "start": 0.0, "end": 0.5},
             {"word": "there", "start": 0.6, "end": 1.0}]
    segs = hf_runner._segments_from_words(words, max_gap=0.4, max_len=10.0)
    assert segs and segs[0]["text"] == "hello there"
    assert segs[0]["start"] == 0.0 and segs[0]["end"] == 1.0
