# asr-bench v0.2 — Information-theoretic metrics + multi-transcript fusion

**Status:** Approved design (2026-06-01)
**Author:** Kevin Rank (Ryfter) + Claude Code
**Supersedes/extends:** v0.1 (Whisper-only via faster-whisper), `SPEC.md` roadmap

## Motivation

Two related upgrades to asr-bench, driven by:

1. **The Morris/Maier/Green paper** *"From WER and RIL to MER and WIL: improved
   evaluation measures for connected speech recognition."* WER measures *edit
   cost* (and can exceed 100%); for almost every use beyond a dictation machine,
   the proportion of **information communicated** is the more meaningful score.
   The paper introduces two bounded-[0,1] measures from the same H/S/D/I counts:
   - **MER** (Match Error Rate) = `(S+D+I) / (H+S+D+I)` — probability a given
     word-match is an error.
   - **WIL** (Word Information Lost) = `1 − H²/(N₁·N₂)` where N₁=ref words,
     N₂=hyp words — proportion of word information lost.
   - WIP (Word Information Preserved) = `1 − WIL`.

2. **A multi-transcript fusion capability.** Combine multiple ASR outputs
   (Whisper turbo/large/medium) **plus** the Panopto editorial caption into a
   single best transcript, using an LLM for contextual judgment (e.g. picking
   "AI" over "I", normalizing "9 to 11" → "9:00–11:00 am", rejecting
   nonsensical readings using known schedule/context).

### Levenshtein background (for the record)

All three metrics ride on **word-level Levenshtein (edit) distance**: the minimum
number of insert/delete/substitute edits to turn the reference token sequence
into the hypothesis. Computed by dynamic programming over an (N₁+1)×(N₂+1)
matrix; backtracking yields the H/S/D/I counts and the alignment. `jiwer`
(already a dependency) computes this via RapidFuzz's C++ Levenshtein and exposes
`wer`, `mer`, `wil`, `wip`, and `process_words` (counts + alignment) from a
single call. The symblai/speech-recognition-evaluation tool computes a strict
subset (WER, WIL, Levenshtein, color diff); its only feature we lack is the
visual alignment diff, which `jiwer.visualize_alignment` provides.

---

## Part A — Richer metrics

### Behavior

- Replace the standalone `jiwer.wer()` call (in both `FasterWhisperEngine` and
  `NimEngine`) with a single `jiwer.process_words(ref_norm, hyp_norm)` call,
  reading WER, MER, WIL, and the H/S/D/I counts off the returned `WordOutput`.
- `ClipResult` gains: `mer`, `wil`, `hits`, `substitutions`, `deletions`,
  `insertions`. `ModelResult` gains `avg_mer`, `avg_wil` (means over clips,
  same nan-handling as `avg_wer`).
- **Report:**
  - Headline + per-model tables gain **MER%** and **WIL%** columns beside WER%.
  - Per-clip table gains **S / D / I** columns (raw edit ops).
  - All three metrics carry the existing gold/proxy reference label.
- **`--show-alignment`** (default off): writes per-clip inline alignment diff
  (`jiwer.visualize_alignment`) into the report (or a sidecar file if verbose).

### Tests (Part A)

The paper's **Table 1** provides exact oracle vectors. In the paper's notation,
lowercase `i` is an inserted hypothesis word and lowercase `d` is a deleted
reference word; the table below expands each to concrete token lists (values are
percentages, rounded to the paper's integers):

| paper ref/hyp | reference tokens | hypothesis tokens | H,S,D,I | WER | MER | WIL |
|---|---|---|---|---|---|---|
| `X` / `X` | `[a]` | `[a]` | 1,0,0,0 | 0 | 0 | 0 |
| `Xiii` / `XXYY` | `[a]` | `[a, a, b, b]` | 1,0,0,3 | 300 | 75 | 75 |
| `XYX` / `XZd` | `[a, b, a]` | `[a, c]` | 1,1,1,0 | 67 | 67 | 83 |
| `X` / `Y` | `[a]` | `[b]` | 0,1,0,0 | 100 | 100 | 100 |
| `Xi` / `YZ` | `[a]` | `[b, c]` | 0,1,0,1 | 200 | 100 | 100 |

Verification of the formulas against row 3 (`XYX`/`XZd`): errors `S+D+I = 2`,
`N₁ = 3` → WER `2/3 = 67%`; MER `2/(1+1+1+0) = 67%`; WIL `1 − 1²/(3·2) = 83%`.

---

## Part B — Fusion stage

A **post-processing pass** that runs *after* the normal benchmark, consuming the
transcripts it already produced. Triggered by `--fuse`. Does **not** implement
the `Engine` ABC (engines transcribe audio independently; fusion depends on other
engines' outputs).

### Inputs per clip

- Each model's VTT cues (timed) — already written next to source audio in v0.1.
- Panopto's caption file (timed), via the existing reference path.
- The context file + glossary (`--context`, optional `--glossary`).

### Windowing (shared by both profiles)

- Walk the clip timeline in `--window` spans (default **25s**) with `--overlap`
  (default **5s**) — overlap carries context across boundaries (RAG-style).
- For each window, collect every timed source's text whose cues overlap the
  window's time range. Untimed sources are text-aligned into the window as a
  fallback.
- Timing anchor for window boundaries: `--fuse-base` (default
  **`large-v3-turbo`**).

### Profiles (`--profile verbatim|kb|both`, default `both`)

Same windowing machinery; differ only in prompt constraints and output shape.

**`verbatim`** (accessibility captions + scoring reference)
- Prompt constraint: *restore the actually-spoken words; choose between
  homophones/near-homophones (e.g. "AI" vs "I") using context; do NOT rephrase,
  do NOT clean up grammar, preserve disfluencies.*
- Panopto's role: spelling / proper-noun hints only (it editorializes wording).
- Output: `<base>_Captions_Fused.vtt` next to source audio. Overlap regions are
  reconciled (each output cue assigned to exactly one window, by cue midpoint).
- **Eligible as an improved scoring reference** (consensus of what was said).

**`kb`** (knowledge base / RAG)
- Prompt constraint: *rewrite for clarity; normalize times/numbers/dates; fix
  references using context; produce clean readable prose.*
- Panopto's role: strong — it is the readability exemplar.
- Output: overlapping **time-tagged chunks** as `<base>_KB_Fused.jsonl` (each:
  `{start, end, text}`) — overlap retained (it's a feature for RAG) — plus a
  human-readable `<base>_KB_Fused.md`.
- **Never** used as a WER reference (rephrasing would unfairly penalize
  literal-but-correct models).

### Drift guard

Per window, compute WER(fused vs base-model text). Flag windows diverging past a
threshold (default configurable) as possible hallucination/omission. Flagged
windows are surfaced in the report's fusion section, not silently trusted.

### Re-scoring (`--rescore-against-fused`, optional)

Re-run Part A metrics for every model against the **verbatim** fused VTT. Emit a
**second metrics table** explicitly labeled:

> *reference = fused verbatim consensus (agreement-biased — scores favor models
> that agreed with the majority; treat as relative, not absolute)*

so the circularity is never hidden.

### Pluggable LLM backend (`LLMBackend` ABC)

Mirrors the `Engine` ABC pattern. Method: `fuse(window_payload, profile,
context, glossary) -> str`.

- **`OllamaBackend`** (default) — local HTTP (`http://localhost:11434`), offline,
  free. Default model `qwen2.5`.
- **`CliBackend`** — shells out to an authenticated frontier CLI
  (`--llm cli:claude`, `--llm cli:"gemini -p"`, etc.). Prompt passed via
  stdin/temp-file to avoid arg-length limits. Uses an existing subscription;
  **no API key stored in asr-bench**.
- **`ApiBackend`** — stub for later (v0.5+ cloud line).
- **`FakeLLMBackend`** — deterministic, test-only; lets the entire fusion path
  be tested without a live LLM (same strategy used for the NIM engine).
- Selected via `--llm <backend>:<spec>` (e.g. `ollama:qwen2.5`, `cli:claude`).

### Context scaffolding (`--init-context [path]`)

Writes a guided `context.md` template (with a glossary section) containing
commented prompts suggesting what to fill in:

- **Schedule & recurring times** (e.g. "I teach 9–11am; no evening classes")
- **People / names** (proper spellings)
- **Acronyms & jargon with canonical spellings** (e.g. "AI, not I")
- **Domain / course terms**
- **Known mishearings to watch for**
- **Style preferences** (caption vs KB)

User edits it, then passes `--context context.md` (and optionally a separate
`--glossary` file that overrides the in-context glossary section).

---

## CLI summary (new flags)

```
--show-alignment                 # Part A: per-clip alignment diff (default off)
--fuse                           # Part B: enable fusion stage
--profile verbatim|kb|both       # default both
--fuse-base <model>              # timing anchor (default large-v3-turbo)
--llm <backend>:<spec>           # default ollama:qwen2.5
--context <file>                 # context/glossary file
--glossary <file>                # optional separate glossary override
--window <seconds>               # default 25
--overlap <seconds>              # default 5
--rescore-against-fused          # re-score models vs verbatim fused reference
--init-context [path]            # scaffold a context.md template and exit
```

---

## Code organization

- **Stays single-file** in `asr_bench.py` per the project rule, organized with
  clear `# ---- Fusion ----` / `# ---- LLM backends ----` sections. The
  `engines/` package split remains deferred (it triggers when WhisperX/NeMo
  land). Flag during implementation if fusion genuinely outgrows single-file.

## Testing

- **Metrics:** the five Table-1 oracle vectors above.
- **Windowing/overlap:** correct cue grouping by time range; overlap inclusion.
- **VTT stitching:** overlapping windows reconcile to non-overlapping output
  cues (midpoint assignment); timestamps preserved.
- **KB chunks:** overlapping time-tagged JSONL emitted; `.md` rendered.
- **Drift guard:** flags a deliberately divergent fused window.
- **Profile prompts:** verbatim vs kb produce distinct constraints.
- **Backends:** `CliBackend` subprocess mocked; `OllamaBackend` HTTP mocked;
  `FakeLLMBackend` drives an end-to-end fusion test.
- **`--init-context`:** writes a non-empty template with all sections.

## Documentation updates

- `CLAUDE.md` — Status table → v0.2; new workflows; new flags.
- `SPEC.md` — v0.2 line (metrics + fusion), ahead of the prior WhisperX slot;
  decision-log entries.
- `README` — metrics explanation; fusion usage; profile distinction.
- New memory note for any pending validation (e.g. live-LLM fusion run).

## Documented caveats (must appear in output/docs)

1. **Consensus-reference bias** — `--rescore-against-fused` uses a reference
   built from the models, so scores favor agreement; relative, not absolute.
2. **LLM-hallucination risk** — mitigated but not eliminated by the drift guard;
   verbatim output should be spot-checked before use as a gold reference.
3. **Accessibility** — only the **verbatim** profile targets ADA/WCAG caption
   fidelity. The **kb** profile rephrases and is **not** compliant captions.

## Sequencing

One coherent v0.2 feature shipped together. Implementation order within it:
Part A (metrics) first (small, low-risk, and the fusion re-scoring path depends
on it), then Part B (fusion) — windowing → LLM backends → profiles → assembly →
drift guard → re-scoring → context scaffolding → docs.

## Out of scope (YAGNI)

- CER and character-level metrics (paper is word-level).
- Direct cloud API integration beyond the `ApiBackend` stub.
- ROVER/voting fusion (the qualitative LLM approach is the chosen path).
- Diarization / speaker labels (still the v0.2-WhisperX line in `SPEC.md`).
