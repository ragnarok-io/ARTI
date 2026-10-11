"""Trusted application entrypoint reusing native prediction/validation journals."""

from pathlib import Path

import arti


def predict(context):
    options = context.options
    profile = options.get("files")
    files = None if profile is None else arti.PredictionFilesConfig.from_dict(profile)
    episodes = None if options.get("episodes") is None else arti.EpisodeRunConfig.from_dict(options["episodes"])
    run = arti.PredictionRunConfig(
        context.outputs["results"], batch_size=options.get("batch_size"), files=files, episodes=episodes
    )
    return arti.run_prediction(
        context.package,
        context.inputs["requests"],
        run,
        resume=options.get("resume", False),
        device=options.get("device", "cpu"),
        task_factory=options["task_factory"],
    )


def validate(context):
    options = context.options
    files = (
        None
        if options.get("files") is None
        else arti.PredictionFilesConfig.from_dict(options["files"])
    )
    episodes = None if options.get("episodes") is None else arti.EpisodeRunConfig.from_dict(options["episodes"])
    return arti.validate_recipe_deployment(
        context.package,
        context.inputs["requests"],
        arti.PredictionRunConfig(
            context.outputs["results"], batch_size=options.get("batch_size"), files=files, episodes=episodes
        ),
        arti.EvaluationSuiteConfig.read(context.inputs["suite"]),
        targets=context.inputs.get("references"),
        task_factory=options["task_factory"],
        resume=options.get("resume", False),
        output=context.outputs["evaluation"],
    )


def describe(context):
    import os

    return {
        "cwd": str(Path.cwd()),
        "package": str(context.package),
        "pid": os.getpid(),
        "variables": {
            name: os.environ.get(name) for name in context.options.get("environment_keys", ())
        },
    }


def fail(context):
    raise RuntimeError("application execution failed")


def wait(context):
    import time

    Path(context.outputs["marker"]).write_text("started", encoding="utf-8")
    time.sleep(context.options["seconds"])
