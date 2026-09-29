# Pre-collected Qwen3-30B-A3B routing traces

## Archive

- File: `ther_traces_qwen3_30b_a3b.tar.xz`
- Size: 1202919576 bytes (~1.12 GiB)
- SHA256: `64612e7d5256436174d23b753987d387574b7d7c179d6dc579c8fc7439409f7d`
- Manifest: `TRACE_MANIFEST.csv`
- Checksums: `SHA256SUMS`

If the archive is absent, `./ther/scripts/setup_traces.sh` downloads it from
Zenodo (`https://zenodo.org/records/21888851`).

From the artifact root:

```bash
./ther/scripts/setup_traces.sh
./ther/scripts/run_quick.sh
```

Archive layout:

```text
traces/<workload>/raw_trace.h5
traces/<workload>/metadata.json
traces/<workload>/requests.json
...
```

Workloads: mixed, arxiv, pubmed_central, github, stackexchange, wikipedia,
freelaw, hackernews, pile_cc.

## Note on redundant compressions

Individual `*.raw_trace.h5.xz` files may also appear in this directory from
development packaging. The **canonical** release object for reviewers is the
single `ther_traces_qwen3_30b_a3b.tar.xz` archive.
