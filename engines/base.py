"""Shared, torch-free support layer for every Engine family: the Engine ABC, the
result/config dataclasses, metrics, VRAM sampling, audio decode, and the VTT/words
writers. Engine modules import from here; asr_bench re-exports these names so the
public surface (import asr_bench / python asr_bench.py) is unchanged.

Torch-free at module scope by contract -- any heavy import belongs inside an
engine's run()/adapter, not here."""
