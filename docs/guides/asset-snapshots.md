# Native Asset Snapshots

Keep application records, binary files and native tensors distinct. A
`DatasetSnapshot` copies an explicitly declared local file closure. It does not
decode media, infer labels, assign splits, impose a tensor rank, or translate a
task to tokens. The source is immutable by convention; content verification
detects changed/missing files, not concurrent source transactions.

## Declare Records And Assets

```python
snapshot = arti.create_dataset_snapshot(
    arti.DatasetSnapshotConfig(
        records="records.jsonl",
        assets={"input-a": "input.bin", "reference-a": None},
        bindings={
            "sample-a": [
                arti.AssetReference("input-a", role="input"),
                arti.AssetReference("reference-a", role="reference", required=False),
            ],
        },
        asset_metadata={"input-a": {"encoding": "my-array-format"}},
        source_revision="my-source-revision",
        id_field="id", split_field="split", group_field="entity",
    ),
    "outputs/snapshot",
)
print(snapshot.manifest()["inventory"])
rows = list(snapshot.records("train"))
input_path = snapshot.asset("input-a")
```

JSONL records retain their original values and ordering. IDs are unique strings.
Splits are application names; no train/validation split is invented. Use
`split_field=None` for unsplit inference records. Labels are not required by the
snapshot. A declared group cannot appear in two splits; omit `group_field` when
that is not your application's grouping rule. Other time/relation metadata can
remain in the original records or declared JSON metadata.

Assets have logical identities independent of source paths. Each asset must be
referenced by a record; input/reference/other roles are explicit caller values.
An absent optional asset is declared as `None`, retains `available=False`, and
is not synthesized into evidence. Required unavailable assets are errors.
Named logical aliases of identical bytes share one copied blob. The inventory
counts references, logical assets, unique contents/bytes, referenced bytes and
unavailable references before expensive decoding.

The directory contains `records.jsonl`, content-addressed `assets/` files and
`dataset.json`. It atomically appears only after copying and checksumming the
closure. The manifest pins source revision, metadata, original records, bindings
and content; it contains no automatic host source paths. Move the entire
directory and reopen `DatasetSnapshot(new_path)` without changing its identity.
Do not put relocatable paths inside semantic record/metadata fields yourself.
Inspection verifies bytes without importing a task factory or decoding tensors.

## Batch And Reuse Preparation

The application supplies one installed, deterministic asset-local transform:

```python
def prepare_arrays(requests, *, options):
    results = []
    for request in requests:
        tensor = decode_my_format(request.source, request.metadata)
        write_my_tensor(request.directory / "tensor.bin", tensor, options)
        results.append({"shape": list(tensor.shape), "encoding": "my-tensor-format"})
    return results

prepared = arti.prepare_dataset_assets(
    snapshot, "outputs/cache",
    arti.AssetTransformConfig(
        "my_package.transforms:prepare_arrays",
        revision="my-decoder-revision",
        options={"sampling": "my-declared-policy"},
    ),
    roles=("input",), batch_size=16,
)
tensor_files = prepared.files("input-a")
training_data_identity = prepared.manifest()
print(prepared.report)
```

The callback receives a batch of `AssetPreparationInput` objects: source path,
output directory, source content digest and source metadata. It produces files
inside each supplied directory and returns one JSON metadata object per source.
It can vectorize decoding across the batch. ARTI streams file checksums/copies;
it never embeds binary payloads in JSON. `entry.json` is reserved for the receipt.

Cache keys bind source bytes, source metadata, installed factory reference and
kwargs, caller-declared transform revision and options. Change the revision when
code/decoder/external dependencies change. This declaration is not automatic
code-byte fingerprinting. Outputs must depend only on those inputs, not source
paths, peer records/batches, undeclared RNG or mutable external state. Put any
sampling seed in options. Training-time random augmentation, record alignment,
collation and samplers can remain native TaskRecipe data-loader behavior;
do not cache a context-dependent transform as if it were asset-local.

Each complete source output closure is atomically committed to
`cache/entries/<key>/`. A retry verifies committed output files and prepares only
missing keys. Fully cached calls do not even import the transform. `produced`,
`reused`, `batches` and `unique_sources` refer to physical per-key work, not
logical reference counts. `stop_requested` is checked before another batch;
incomplete preparation cannot supply `manifest()` for training. `on_batch` runs
after publication. Failed callbacks/publish attempts leave previously committed
entries intact. This is one local writer, not a distributed cache or global
exactly-once callback execution guarantee.

Move snapshot and cache independently and call preparation with the new paths.
Semantic manifests omit runtime counters/cache paths, so the same input and
transform retain dataset identity. Select roles explicitly to keep evaluator
reference files outside the input preparation path.

## Connect The Existing Workflow

Use snapshot/preparation manifests in the existing `TaskRecipe.dataset_manifest`
and supply `data_preparation=prepared.report` for durable run diagnostics. The
native dataset, collation, sampler, objective and batch weights remain yours.
Keep snapshot/cache locations in `TaskRecipeConfig.resources`, not options.
Existing qualified checkpoint recovery checks the same task/data identity;
resource relocation does not alter it. Existing portable task packaging can
include explicitly selected snapshot/cache directories as named inference
resources. That does not authorize distributing private training labels.

`examples/asset_task_recipe.py` demonstrates application-owned binary float
decoding, vector/matrix inputs, native ARTI model buffers and regression loss.
Prediction reads only input-role arrays; it does not need targets in requests.
The example factory/decoder is caller code, not an auto-installed core adapter.

## CLI

```bash
arti recipe snapshot-data snapshot.json --output outputs/snapshot
arti recipe inspect-data outputs/snapshot --output outputs/data-inspection.json
arti recipe prepare-assets outputs/snapshot --transform transform.json --cache-dir outputs/cache --role input --batch-size 16 --output outputs/preparation.json
```

Snapshot JSON/TOML configurations use the Python field names above, with each
binding represented as `{asset_id, role, required}`. Source file paths resolve
relative to the configuration file. Transform configurations use
`factory`, `revision`, `options`, optionally
`format="arti.asset-transform-config"`. Inspection reports live outside the
immutable snapshot; preparation reports must not overwrite cache entries.

## Design References

[HF Datasets cache fingerprints](https://huggingface.co/docs/datasets/main/en/about_cache)
bind source state and transforms; [DVC content-addressed data](https://doc.dvc.org/user-guide/project-structure/internal-files)
separates logical paths from unique stored content. ARTI transfers these identity
and reuse principles, without adding either engine, inferred code serialization,
remote storage or installation as a dependency. Checked 2026-10-11.
