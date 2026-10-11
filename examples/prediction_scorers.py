"""Model-independent scorers; no model/task factory or training code is imported."""

import math


def regression(pairs, *, prediction_field="prediction", target_field="target"):
    scores = []
    for pair in pairs:
        predicted = pair["prediction"][prediction_field]
        target = pair["target"][target_field]
        if any(
            type(value) not in {int, float} or not math.isfinite(value)
            for value in (predicted, target)
        ):
            raise ValueError("regression scoring requires finite original-unit scalar values")
        scores.append(
            {
                "id": pair["id"],
                "squared_error": (predicted - target) ** 2,
                "absolute_error": abs(predicted - target),
            }
        )
    return {
        "metrics": {
            "mse": sum(row["squared_error"] for row in scores) / len(scores),
            "mae": sum(row["absolute_error"] for row in scores) / len(scores),
        },
        "scores": scores,
    }


def classification(pairs, *, prediction_field="label", target_field="label"):
    scores = [
        {
            "id": pair["id"],
            "correct": pair["prediction"][prediction_field] == pair["target"][target_field],
        }
        for pair in pairs
    ]
    return {
        "metrics": {"accuracy": sum(row["correct"] for row in scores) / len(scores)},
        "scores": scores,
    }
