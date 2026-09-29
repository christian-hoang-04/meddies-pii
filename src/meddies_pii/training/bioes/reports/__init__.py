"""Static BIOES training report internals."""

from meddies_pii.annotations.span_rendering import highlight_spans, snippet_html

from .data_shape import render_data_shape_report_html, write_data_shape_report
from .eval_prediction_audit import (
    metric_scenarios,
    read_prediction_records,
    render_eval_prediction_audit_report,
    write_adjudication_batches,
    write_eval_prediction_audit_outputs,
)
from .html import render_training_report_html
from .inference_preview import (
    DEFAULT_INFERENCE_JSON,
    DEFAULT_RESULT_JSON,
    compare_spans,
    preview_metrics,
    render_inference_preview_html,
    write_inference_preview_report,
)
from .inference_preview import (
    DEFAULT_OUTPUT_DIR as DEFAULT_INFERENCE_PREVIEW_OUTPUT_DIR,
)
from .models import LabelExample, RowPreview, TrainingReportOptions, TrainingScan
from .rejection_audit import (
    build_rejection_audit_html,
    load_rejection_records,
    write_rejection_audit_report,
)
from .scanning import (
    scan_adversarial_jsonl,
    scan_audit_jsonl,
    scan_labeled_jsonl,
    scan_training_jsonl,
)
from .span_filter_review import (
    collect_span_filter_review,
    render_span_filter_review_html,
    write_span_filter_review_report,
)
from .training_data_breakdown import (
    TrainingDataSourceConfig,
    TrainingDataSourceStats,
    render_training_data_breakdown_html,
    scan_training_data_source,
    write_training_data_breakdown_report,
)
from .training_setup_preview import render_html as render_training_setup_preview_html

__all__ = [
    "DEFAULT_INFERENCE_JSON",
    "DEFAULT_INFERENCE_PREVIEW_OUTPUT_DIR",
    "DEFAULT_RESULT_JSON",
    "LabelExample",
    "RowPreview",
    "TrainingDataSourceConfig",
    "TrainingDataSourceStats",
    "TrainingReportOptions",
    "TrainingScan",
    "build_rejection_audit_html",
    "collect_span_filter_review",
    "compare_spans",
    "highlight_spans",
    "load_rejection_records",
    "metric_scenarios",
    "preview_metrics",
    "read_prediction_records",
    "render_data_shape_report_html",
    "render_eval_prediction_audit_report",
    "render_inference_preview_html",
    "render_span_filter_review_html",
    "render_training_data_breakdown_html",
    "render_training_report_html",
    "render_training_setup_preview_html",
    "scan_adversarial_jsonl",
    "scan_audit_jsonl",
    "scan_labeled_jsonl",
    "scan_training_data_source",
    "scan_training_jsonl",
    "snippet_html",
    "write_adjudication_batches",
    "write_data_shape_report",
    "write_eval_prediction_audit_outputs",
    "write_inference_preview_report",
    "write_rejection_audit_report",
    "write_span_filter_review_report",
    "write_training_data_breakdown_report",
]
