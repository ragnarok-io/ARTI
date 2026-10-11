"""Neutral application layouts over existing native outputs; never infer."""

import csv
import io
import json


def write_results(context, items):
    results = []
    value_field = context.options.get("value_field", "prediction")
    for index, item in enumerate(items):
        directory = context.directory / context.options.get("folder", "records")
        directory.mkdir(exist_ok=True)
        path = directory / f"{index}.json"
        path.write_text(
            json.dumps({"id": item["id"], "value": item["prediction"][value_field]}),
            encoding="utf-8",
        )
        results.append(
            {"id": item["id"], "files": {"record": path.relative_to(context.directory).as_posix()}}
        )
    return results


def write_table(context, items):
    path = context.directory / "results.csv"
    value_field = context.options.get("value_field", "prediction")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("id", "value"))
        writer.writerows((item["id"], item["prediction"][value_field]) for item in items)
    return [{"id": item["id"], "files": {"table": "results.csv"}} for item in items]


def validate_results(results, *, directory):
    for row in results:
        file = next(iter(row["files"].values()))
        path = directory / file["file"]
        if path.suffix == ".json":
            if json.loads(path.read_text(encoding="utf-8"))["id"] != row["id"]:
                raise ValueError("application layout record ID differs")
        else:
            values = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))
            if row["id"] not in {value["id"] for value in values}:
                raise ValueError("application table does not cover the sample")
    return {"ok": True, "records": len(results)}
