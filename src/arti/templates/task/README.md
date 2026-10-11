# Application-owned file task

ARTI and PyTorch must already be installed in the active Python environment.
This project uses neutral generated vectors and a native ARTI/PyTorch objective.
It downloads nothing and does not install dependencies or submit results.

Run from this directory:

```sh
python -m arti task check project.json
python workflow.py prepare
python workflow.py prepare
python workflow.py train --stop-after 2
python -m arti task status project.json
python -m arti task explain project.json
python workflow.py recover
python workflow.py package
python workflow.py predict --stop-after 1
python workflow.py predict --resume
python workflow.py evaluate
python workflow.py runtime
python workflow.py runtime --resume
python workflow.py export
python -m arti task status project.json
```

`code/task_app.py` owns the model, preprocessing, collation, objective, formats,
scorers, runtime entrypoint and export layout. `workflow.py` calls the existing
ARTI lifecycle APIs; the native training run and prediction journal remain the
state authorities. Train/recover updates the project's selected artifact paths.
Choose a fresh project if you want an independent second training run.

The exported archive preserves results, selected deployment, original inputs,
evaluation and runtime receipts. Formatting never regenerates predictions.
The cold runtime's first invocation uses its own package-qualified journal;
`runtime --resume` reuses that journal without replaying completed predictions.
An offline network hint is not network isolation; strong memory, network and
process-tree enforcement remains unavailable on the built-in local worker.

Read `TUTORIAL.md` for the full installed-package walkthrough, artifact relocation
and a fresh consumer after removal of the original application/data sources.
