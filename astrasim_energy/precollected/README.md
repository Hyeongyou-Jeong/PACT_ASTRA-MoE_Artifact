# Pre-collected Qwen3-30B-A3B routing trace (224 requests)

Real routed-expert trace used by the ASTRA-MoE Energy Evaluation (`astrasim_energy/`). It was
collected using the same trace-collection infrastructure used by the THER evaluation, but it is a
separate trace from the THER policy-evaluation traces and is not part of `ther/`. This directory
holds only checksums and metadata; the archive itself is hosted on Zenodo:
<https://zenodo.org/records/21889194/files/realtrace_qwen30_a3b_224req.tar.xz?download=1>
(record 21889194, DOI 10.5281/zenodo.21889194). Mirror (GitHub release `v3`):
<https://github.com/Hyeongyou-Jeong/PACT_ASTRA-MoE_Artifact/releases/download/v3/realtrace_qwen30_a3b_224req.tar.xz>

- File: `realtrace_qwen30_a3b_224req.tar.xz` (not stored in the source tree)
- Size: 1,872,884,316 bytes compressed (~1.74 GiB); 2,323,767,913 bytes extracted (~2.16 GiB)
- SHA256: see `SHA256SUMS` (first line = archive; remaining lines = archive members)
- Contents and collection commands: `PROVENANCE.md`, `TRACE_MANIFEST.csv`

`astrasim_energy/scripts/setup_realtrace.sh` verifies and extracts the archive. It looks for the
archive in this order: the first argument, `$REALTRACE_ARCHIVE`, then
`astrasim_energy/precollected/realtrace_qwen30_a3b_224req.tar.xz`; if none exists, it downloads the
archive there from Zenodo, falling back to the GitHub mirror (`$REALTRACE_URL` overrides both).
