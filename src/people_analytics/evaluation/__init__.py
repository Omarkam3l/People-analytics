"""Evaluation subsystem."""

from people_analytics.evaluation.detection_metrics import (
    DetectionMetrics,
    compute_iou,
    evaluate_detections_frame,
    evaluate_sequence_detections,
)
from people_analytics.evaluation.tracking_metrics import (
    TrackingMetrics,
    evaluate_tracking_sequence,
)

__all__ = [
    "ConservationCheck",
    "DetectionMetrics",
    "EvaluationFrameState",
    "FullConservationAudit",
    "FullPipelineEvaluator",
    "FullSequenceEvaluationReport",
    "GroundProjectionDiagnostics",
    "ReproducibilityDifference",
    "ReproducibilityReport",
    "SceneEvaluationConfig",
    "TrackingMetrics",
    "TrajectorySummaryStats",
    "compute_iou",
    "evaluate_detections_frame",
    "evaluate_sequence_detections",
    "evaluate_tracking_sequence",
    "generate_conservation_audit_log",
    "generate_markdown_report",
    "get_mot17_02_evaluation_config",
    "get_mot17_09_evaluation_config",
    "get_report_markdown_path",
    "render_visual_artifacts",
    "save_machine_readable_results",
    "verify_reproducibility",
]


def __getattr__(name: str):
    if name == "FullPipelineEvaluator":
        from people_analytics.evaluation.full_pipeline_evaluator import FullPipelineEvaluator
        return FullPipelineEvaluator
    if name in (
        "FullSequenceEvaluationReport",
        "TrajectorySummaryStats",
        "GroundProjectionDiagnostics",
        "ConservationCheck",
        "FullConservationAudit",
        "EvaluationFrameState",
    ):
        from people_analytics.evaluation import models
        return getattr(models, name)
    if name in ("SceneEvaluationConfig", "get_mot17_09_evaluation_config", "get_mot17_02_evaluation_config"):
        from people_analytics.evaluation import scene_config
        return getattr(scene_config, name)
    if name in ("verify_reproducibility", "ReproducibilityReport", "ReproducibilityDifference"):
        from people_analytics.evaluation import reproducibility
        return getattr(reproducibility, name)
    if name in (
        "generate_markdown_report",
        "get_report_markdown_path",
        "generate_conservation_audit_log",
        "save_machine_readable_results",
        "render_visual_artifacts",
    ):
        from people_analytics.evaluation import report_generator
        return getattr(report_generator, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

