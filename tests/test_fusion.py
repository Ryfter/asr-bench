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


def test_write_kb_md(tmp_path):
    audio = tmp_path / "Lecture_default.mp4"
    audio.write_bytes(b"x")
    chunks = [{"start": 0.0, "end": 25.0, "text": "first chunk"}]
    out = asr_bench.write_kb_md(audio, chunks)
    assert out.name == "Lecture_KB_Fused.md"
    body = out.read_text(encoding="utf-8")
    assert "Knowledge base" in body and "first chunk" in body


def test_rescore_against_reference_recomputes_metrics():
    c1 = asr_bench.ClipResult(
        audio="L.mp4", audio_sec=10, transcribe_sec=1, rtfx=10, vram_peak_bytes=None,
        hypothesis="the cat sat", reference_normalized="x", hypothesis_normalized="the cat sat",
        wer=0.9,
    )
    m1 = asr_bench.ModelResult(
        model_id="a", display="A", fw_name="a", params="1", developer="d", languages="en",
        notes="", disk_bytes=None, load_sec=0, clips=[c1],
    )
    ref_cues_by_clip = {"L.mp4": [asr_bench.Cue(0, 10, "the cat sat")]}
    rescored = asr_bench.rescore_against_reference([m1], ref_cues_by_clip)
    assert abs(rescored[0].clips[0].wer) < 1e-9   # perfect match -> 0 WER


def test_rescore_does_not_mutate_original():
    c1 = asr_bench.ClipResult(
        audio="L.mp4", audio_sec=10, transcribe_sec=1, rtfx=10, vram_peak_bytes=None,
        hypothesis="the cat sat", reference_normalized="x", hypothesis_normalized="the cat sat",
        wer=0.9,
    )
    m1 = asr_bench.ModelResult(
        model_id="a", display="A", fw_name="a", params="1", developer="d", languages="en",
        notes="", disk_bytes=None, load_sec=0, clips=[c1],
    )
    asr_bench.rescore_against_reference([m1], {"L.mp4": [asr_bench.Cue(0, 10, "the cat sat")]})
    assert m1.clips[0].wer == 0.9   # original untouched


def test_rescore_unmatched_clip_is_nan():
    import math
    c1 = asr_bench.ClipResult(
        audio="L.mp4", audio_sec=10, transcribe_sec=1, rtfx=10, vram_peak_bytes=None,
        hypothesis="x", reference_normalized="", hypothesis_normalized="x", wer=0.5,
    )
    m1 = asr_bench.ModelResult(
        model_id="a", display="A", fw_name="a", params="1", developer="d", languages="en",
        notes="", disk_bytes=None, load_sec=0, clips=[c1],
    )
    rescored = asr_bench.rescore_against_reference([m1], {})  # no ref for L.mp4
    assert math.isnan(rescored[0].clips[0].wer)


def test_render_fused_rescore_table_labeled_biased():
    c1 = asr_bench.ClipResult(
        audio="L.mp4", audio_sec=10, transcribe_sec=1, rtfx=10, vram_peak_bytes=None,
        hypothesis="hi", reference_normalized="hi", hypothesis_normalized="hi",
        wer=0.0, mer=0.0, wil=0.0,
    )
    m1 = asr_bench.ModelResult(
        model_id="a", display="A", fw_name="a", params="1", developer="d", languages="en",
        notes="", disk_bytes=None, load_sec=0, clips=[c1],
    )
    md = asr_bench.render_fused_rescore_table([m1])
    assert "fused verbatim consensus" in md.lower()
    assert "biased" in md.lower()


def test_run_fusion_stage_end_to_end(tmp_path, monkeypatch):
    audio = tmp_path / "Lecture_default.mp4"
    audio.write_bytes(b"x")
    vtt = tmp_path / "Lecture_Captions_LargeV3Turbo.vtt"
    vtt.write_text("WEBVTT\n\n1\n00:00:00.000 --> 00:00:10.000\nthe AI model\n", encoding="utf-8")

    clip = asr_bench.ClipResult(
        audio="Lecture_default.mp4", audio_sec=10.0, transcribe_sec=1.0, rtfx=10.0,
        vram_peak_bytes=None, hypothesis="the AI model", reference_normalized="",
        hypothesis_normalized="the ai model", wer=0.0, vtt_path=str(vtt),
    )
    mr = asr_bench.ModelResult(
        model_id="large-v3-turbo", display="Whisper Large V3 Turbo", fw_name="large-v3-turbo",
        params="809M", developer="OpenAI", languages="99", notes="", disk_bytes=None,
        load_sec=0.0, clips=[clip],
    )
    pair = asr_bench.Pair(audio=audio, reference=vtt)

    backend = asr_bench.FakeLLMBackend(lambda prompt: "the AI model")
    fusion_md, rescored = asr_bench.run_fusion_stage(
        results=[mr], pairs=[pair], backend=backend,
        profiles=["verbatim", "kb"], base_label="large-v3-turbo",
        context="", glossary="", window=25.0, overlap=5.0, drift_threshold=2.0,
        rescore=True,
    )
    assert (tmp_path / "Lecture_Captions_Fused.vtt").exists()
    assert (tmp_path / "Lecture_KB_Fused.jsonl").exists()
    assert "Fusion" in fusion_md
    assert rescored is not None
