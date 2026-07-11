"""Gradle-like model adaptation entrypoints."""

from .artifacts import ARTIFitResult, AdapterArtifactManifest, BuildTaskSpec, FitReportSummary, FitTaskRecord, ForwardProfile, MechanismSummary, ParameterSummary, create_build_lock, create_deployment_manifest, create_task_graph_payload, plan_provenance_fingerprint, validate_artifact, validate_artifact_payload, validate_build_lock, validate_deployment_manifest, validate_plan, validate_plan_payload, validate_task_graph, validate_task_graph_payload, write_task_graph_artifact
from .batch_schema import BatchSchema, TensorField, attention_mask_to_visibility, infer_batch_schema
from .config import FitProjectConfig, MechanismOverrides, apply_mechanism_overrides, load_fit_config, resolve_fit_config_mechanism, template_fit_config, validate_fit_config, write_fit_config_template
from .doctor import BackendCapabilities, backend_capabilities, doctor_report, doctor_report_markdown, validate_backend_capabilities, write_doctor_report
from .docs import check_fit_config_schema, check_generated_docs, check_task_graph_schema, generate_capabilities_markdown, generate_fit_config_schema, generate_fit_config_schema_json, generate_task_graph_schema, generate_task_graph_schema_json, packaged_fit_config_schema_json, packaged_task_graph_schema_json, write_fit_config_schema, write_generated_docs, write_task_graph_schema
from .insertion import AdapterInsertionPlan
from .metadata import capabilities, list_plugins, list_profiles, list_scales
from .objectives import infer_objectives, resolve_objectives
from .plugins import FitPlugin, get_plugin
from .project import ARTIProject, apply_adapter, fit, project
from .runtime import RuntimeFieldConfig
from .scanner import InsertionCandidate, ScanReport

__all__ = [
    "ARTIFitResult",
    "AdapterArtifactManifest",
    "BuildTaskSpec",
    "FitReportSummary",
    "FitTaskRecord",
    "ForwardProfile",
    "MechanismSummary",
    "ParameterSummary",
    "AdapterInsertionPlan",
    "ARTIProject",
    "InsertionCandidate",
    "ScanReport",
    "FitPlugin",
    "BatchSchema",
    "TensorField",
    "FitProjectConfig",
    "MechanismOverrides",
    "RuntimeFieldConfig",
    "apply_mechanism_overrides",
    "resolve_fit_config_mechanism",
    "BackendCapabilities",
    "load_fit_config",
    "template_fit_config",
    "validate_fit_config",
    "write_fit_config_template",
    "get_plugin",
    "infer_batch_schema",
    "attention_mask_to_visibility",
    "capabilities",
    "list_profiles",
    "list_scales",
    "list_plugins",
    "resolve_objectives",
    "infer_objectives",
    "plan_provenance_fingerprint",
    "create_build_lock",
    "create_deployment_manifest",
    "create_task_graph_payload",
    "backend_capabilities",
    "doctor_report",
    "doctor_report_markdown",
    "validate_backend_capabilities",
    "write_doctor_report",
    "generate_capabilities_markdown",
    "generate_fit_config_schema",
    "generate_fit_config_schema_json",
    "generate_task_graph_schema",
    "generate_task_graph_schema_json",
    "packaged_fit_config_schema_json",
    "packaged_task_graph_schema_json",
    "write_generated_docs",
    "write_fit_config_schema",
    "write_task_graph_schema",
    "check_generated_docs",
    "check_fit_config_schema",
    "check_task_graph_schema",
    "validate_artifact",
    "validate_artifact_payload",
    "validate_build_lock",
    "validate_deployment_manifest",
    "validate_plan",
    "validate_plan_payload",
    "validate_task_graph",
    "validate_task_graph_payload",
    "write_task_graph_artifact",
    "fit",
    "project",
    "apply_adapter",
]
