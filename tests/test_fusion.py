import asr_bench


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
