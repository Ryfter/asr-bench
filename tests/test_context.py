import asr_bench


def test_init_context_template_has_all_sections():
    t = asr_bench.init_context_template()
    for needle in ["Schedule", "Glossary", "Jargon", "Names", "Style", "mishearings"]:
        assert needle.lower() in t.lower(), needle


def test_load_context_reads_file_and_glossary(tmp_path):
    ctx = tmp_path / "context.md"
    ctx.write_text("I teach 9-11am.\n\n## Glossary\nAI not I\n", encoding="utf-8")
    context_text, glossary_text = asr_bench.load_context(str(ctx), None)
    assert "I teach 9-11am." in context_text
    assert "AI not I" in glossary_text


def test_load_context_separate_glossary_file_overrides(tmp_path):
    ctx = tmp_path / "context.md"
    ctx.write_text("topic notes\n\n## Glossary\nin-file gloss\n", encoding="utf-8")
    gl = tmp_path / "gloss.txt"
    gl.write_text("override gloss", encoding="utf-8")
    _, glossary_text = asr_bench.load_context(str(ctx), str(gl))
    assert glossary_text.strip() == "override gloss"


def test_load_context_none_returns_empty():
    assert asr_bench.load_context(None, None) == ("", "")
