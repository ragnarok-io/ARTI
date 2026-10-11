"""Application-owned multi-view scorers and explicit composite metric rules."""

import json
import struct


def file_validity(pairs, *, directory, reference_directory, files_field):
    scores = []
    for pair in pairs:
        value = pair["prediction"]
        files = value[files_field]
        array = json.loads((directory / files["array"]["file"]).read_text(encoding="utf-8"))
        scalar = struct.unpack("<d", (directory / files["binary"]["file"]).read_bytes())[0]
        valid = array == [value["prediction"]] and scalar == value["prediction"]
        scores.append(
            {
                "id": pair["id"],
                "valid": valid,
                "bytes": sum(file["bytes"] for file in files.values()),
            }
        )
    return {
        "metrics": {"valid_fraction": sum(row["valid"] for row in scores) / len(scores)},
        "scores": scores,
    }


def binary_error(pairs, *, directory, reference_directory, files_field):
    scores = []
    for pair in pairs:
        reference = next(iter(pair["reference_assets"].values()))
        target = struct.unpack("<d", (reference_directory / reference["file"]).read_bytes())[0]
        file = pair["prediction"][files_field]["binary"]
        predicted = struct.unpack("<d", (directory / file["file"]).read_bytes())[0]
        scores.append({"id": pair["id"], "squared_error": (predicted - target) ** 2})
    return {
        "metrics": {"mse": sum(row["squared_error"] for row in scores) / len(scores)},
        "scores": scores,
    }


def aggregate_views(views, *, coefficients):
    metrics, terms = {}, []
    for view, metric, coefficient in coefficients:
        if metric not in views[view]["metrics"]:
            return {"metrics": {}, "missing_view": view}
        terms.append(coefficient * views[view]["metrics"][metric])
    metrics["composite"] = sum(terms)
    metrics["minimum_coverage"] = min(view["coverage"]["total_fraction"] for view in views.values())
    return {"metrics": metrics, "coefficients": coefficients}


def mutate(pairs):
    from examples.prediction_scorers import regression

    result = regression(pairs)
    pairs[0]["input"]["id"] = "modified"
    pairs[0]["target"]["target"] = "modified"
    return result


def fail(pairs):
    raise RuntimeError("application scorer failed")
