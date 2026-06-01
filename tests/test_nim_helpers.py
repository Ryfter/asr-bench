import types
import asr_bench


def test_build_nim_auth_kwargs_insecure():
    kw = asr_bench.build_nim_auth_kwargs("localhost:50051", None, False)
    assert kw["uri"] == "localhost:50051"
    assert kw["use_ssl"] is False
    assert "metadata_args" not in kw or not kw["metadata_args"]


def test_build_nim_auth_kwargs_with_key_enables_ssl_and_bearer():
    kw = asr_bench.build_nim_auth_kwargs("grpc.example.com:443", "ABC123", False)
    assert kw["use_ssl"] is True
    assert ["authorization", "Bearer ABC123"] in kw["metadata_args"]


def test_build_nim_auth_kwargs_explicit_ssl_no_key():
    kw = asr_bench.build_nim_auth_kwargs("host:50051", None, True)
    assert kw["use_ssl"] is True


def _fake_response():
    # Mimic riva RecognizeResponse: results[].alternatives[0].transcript / .words[]
    Word = lambda w, s, e: types.SimpleNamespace(word=w, start_time=s, end_time=e)
    alt = types.SimpleNamespace(
        transcript="hello world. how are you",
        words=[Word("hello", 0, 400), Word("world.", 400, 900),
               Word("how", 1000, 1200), Word("are", 1200, 1400),
               Word("you", 1400, 1700)],
    )
    result = types.SimpleNamespace(alternatives=[alt])
    return types.SimpleNamespace(results=[result])


def test_nim_response_to_hypothesis():
    hyp = asr_bench.nim_response_to_hypothesis(_fake_response())
    assert hyp == "hello world. how are you"


def test_nim_response_to_words_converts_ms_to_seconds():
    words = asr_bench.nim_response_to_words(_fake_response())
    assert words[0] == (0.0, 0.4, "hello")
    assert words[1][2] == "world."


def test_group_words_into_cues_breaks_on_sentence_end():
    words = asr_bench.nim_response_to_words(_fake_response())
    cues = asr_bench.group_words_into_cues(words, max_words=12, max_span=6.0)
    # "hello world." ends a sentence -> first cue closes there
    assert cues[0][2] == "hello world."
    assert cues[0][0] == 0.0
    assert cues[0][1] == 0.9
    assert cues[1][2] == "how are you"


def test_group_words_into_cues_breaks_on_max_words():
    words = [(float(i), float(i) + 0.5, f"w{i}") for i in range(15)]
    cues = asr_bench.group_words_into_cues(words, max_words=5, max_span=999.0)
    assert len(cues) == 3
    assert cues[0][2] == "w0 w1 w2 w3 w4"


def test_group_words_into_cues_empty():
    assert asr_bench.group_words_into_cues([], 12, 6.0) == []


def test_group_words_into_cues_breaks_on_max_span():
    # No sentence punctuation, few words, but the running span reaches max_span.
    words = [(0.0, 1.0, "a"), (3.0, 4.0, "b"), (6.0, 7.0, "c"), (8.0, 9.0, "d")]
    cues = asr_bench.group_words_into_cues(words, max_words=99, max_span=6.0)
    # span hits >=6.0 when "c" lands (end 7.0 - start 0.0), closing the first cue
    assert len(cues) == 2
    assert cues[0] == (0.0, 7.0, "a b c")
    assert cues[1][2] == "d"


def test_nim_response_to_hypothesis_multiple_results():
    import types
    r1 = types.SimpleNamespace(alternatives=[types.SimpleNamespace(transcript="foo")])
    r2 = types.SimpleNamespace(alternatives=[types.SimpleNamespace(transcript="bar baz")])
    resp = types.SimpleNamespace(results=[r1, r2])
    assert asr_bench.nim_response_to_hypothesis(resp) == "foo bar baz"
