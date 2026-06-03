import asr_bench
from pathlib import Path


def _payload():
    return asr_bench.WindowPayload(
        start=0.0, end=25.0,
        sources={
            "large-v3-turbo": "i think AI is great",
            "Panopto": "I think I is great",
        },
        prev_fused="Welcome back everyone.",
    )


def test_verbatim_prompt_forbids_rephrasing():
    p = asr_bench.build_fusion_prompt(_payload(), "verbatim", context="I teach 9-11am.", glossary="AI not I")
    low = p.lower()
    assert "verbatim" in low or "do not rephrase" in low or "actually" in low
    assert "AI not I" in p              # glossary injected
    assert "I teach 9-11am." in p       # context injected
    assert "large-v3-turbo" in p        # sources injected
    assert "Welcome back everyone." in p  # carryover context


def test_kb_prompt_allows_rewriting():
    p = asr_bench.build_fusion_prompt(_payload(), "kb", context="", glossary="")
    low = p.lower()
    assert "rewrite" in low or "clarity" in low or "readable" in low


def test_prompts_differ_by_profile():
    a = asr_bench.build_fusion_prompt(_payload(), "verbatim", "", "")
    b = asr_bench.build_fusion_prompt(_payload(), "kb", "", "")
    assert a != b


# ---- fuse_clip orchestrator tests -------------------------------------------

def _sources():
    turbo = [asr_bench.Cue(0, 10, "the AI model learns"), asr_bench.Cue(10, 20, "we meet nine to eleven"), asr_bench.Cue(20, 30, "no class tonight")]
    panopto = [asr_bench.Cue(0, 10, "the I model learns"), asr_bench.Cue(10, 20, "we meet 9 to 11"), asr_bench.Cue(20, 30, "no class tonight")]
    return {"large-v3-turbo": turbo, "Panopto": panopto}


def test_fuse_clip_verbatim_produces_cues():
    backend = asr_bench.FakeLLMBackend(lambda prompt: "FUSEDTEXT")
    res = asr_bench.fuse_clip(
        duration=30.0, base_label="large-v3-turbo", sources=_sources(),
        profiles=["verbatim"], backend=backend, context="", glossary="",
        window=25.0, overlap=5.0, drift_threshold=2.0,
    )
    assert res.verbatim_cues, "expected verbatim cues"
    assert all(c.text == "FUSEDTEXT" for c in res.verbatim_cues)
    for a, b in zip(res.verbatim_cues, res.verbatim_cues[1:]):
        assert b.start >= a.end - 1e-6


def test_fuse_clip_kb_chunks_retain_overlap_spans():
    backend = asr_bench.FakeLLMBackend(lambda prompt: "kb chunk")
    res = asr_bench.fuse_clip(
        duration=60.0, base_label="large-v3-turbo", sources=_sources(),
        profiles=["kb"], backend=backend, context="", glossary="",
        window=25.0, overlap=5.0, drift_threshold=2.0,
    )
    assert len(res.kb_chunks) >= 2
    assert res.kb_chunks[0]["text"] == "kb chunk"
    assert "start" in res.kb_chunks[0] and "end" in res.kb_chunks[0]


def test_drift_guard_flags_divergent_window():
    backend = asr_bench.FakeLLMBackend(lambda prompt: "zzz qqq xyz")
    res = asr_bench.fuse_clip(
        duration=10.0, base_label="large-v3-turbo", sources=_sources(),
        profiles=["verbatim"], backend=backend, context="", glossary="",
        window=25.0, overlap=5.0, drift_threshold=0.5,
    )
    assert res.flags, "expected a drift flag"


def test_write_fused_vtt(tmp_path):
    audio = tmp_path / "Lecture_default.mp4"
    audio.write_bytes(b"x")
    cues = [asr_bench.Cue(0.0, 5.0, "hello"), asr_bench.Cue(5.0, 10.0, "world")]
    out = asr_bench.write_fused_vtt(audio, cues)
    assert out.name == "Lecture_Captions_Fused.vtt"
    body = out.read_text(encoding="utf-8")
    assert "WEBVTT" in body and "hello" in body


def test_write_kb_jsonl(tmp_path):
    import json
    audio = tmp_path / "Lecture_default.mp4"
    audio.write_bytes(b"x")
    chunks = [{"start": 0.0, "end": 25.0, "text": "a"}, {"start": 20.0, "end": 45.0, "text": "b"}]
    out = asr_bench.write_kb_jsonl(audio, chunks)
    assert out.name == "Lecture_KB_Fused.jsonl"
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert rows[1]["text"] == "b"
