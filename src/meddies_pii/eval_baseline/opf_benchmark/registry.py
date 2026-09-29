from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from meddies_pii.eval_baseline.opf_benchmark.results import BenchmarkResultRow

PathId = Literal["A", "A2", "B", "C"]
CompileMode = Literal["none", "reduce-overhead", "max-autotune"]
TritonMode = Literal["on", "off", "na"]
OnnxVariant = Literal["none", "fp32", "fp16", "q4", "q4f16", "quantized"]
AttentionKernel = Literal["native-banded", "default", "sdpa", "flash_attention_2", "onnx-runtime"]
LengthBucketName = Literal["short", "medium", "long"]

PATH_IDS: tuple[PathId, ...] = ("A", "A2", "B", "C")
BATCH_SIZES: tuple[int, ...] = (1, 8, 16, 32, 64, 128, 256)
COMPILE_MODES: tuple[CompileMode, ...] = ("none", "reduce-overhead", "max-autotune")
TRITON_MODES: tuple[Literal["on", "off"], ...] = ("on", "off")
LENGTH_BUCKET_NAMES: tuple[LengthBucketName, ...] = ("short", "medium", "long")
ONNX_VARIANTS: tuple[Literal["fp32", "fp16", "q4", "q4f16", "quantized"], ...] = (
    "fp32",
    "fp16",
    "q4",
    "q4f16",
    "quantized",
)
B_DTYPES: tuple[Literal["bf16", "fp16"], ...] = ("bf16", "fp16")
ATTENTION_KERNELS: tuple[Literal["default", "sdpa", "flash_attention_2"], ...] = (
    "default",
    "sdpa",
    "flash_attention_2",
)

PATH_NAMES: dict[PathId, str] = {
    "A": "native-triton-crf-batch1",
    "A2": "native-triton-batched-crf",
    "B": "hf-automodel-argmax-bioes",
    "C": "onnx-runtime-argmax",
}
ONNX_DTYPE: dict[str, str] = {
    "fp32": "fp32",
    "fp16": "fp16",
    "q4": "int4",
    "q4f16": "int4-fp16",
    "quantized": "int8",
}


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    path_id: PathId
    path_name: str
    batch_size: int
    length_bucket: LengthBucketName
    dtype: str
    compile_mode: CompileMode
    triton: TritonMode
    onnx_variant: OnnxVariant
    attention_kernel: AttentionKernel
    skip_reason: str | None = None

    @property
    def config_id(self) -> str:
        parts = [
            self.path_id,
            f"len-{self.length_bucket}",
            f"bs{self.batch_size}",
            f"dtype-{self.dtype}",
            f"compile-{self.compile_mode}",
            f"triton-{self.triton}",
            f"kernel-{self.attention_kernel}",
        ]
        if self.onnx_variant != "none":
            parts.append(f"onnx-{self.onnx_variant}")
        return "__".join(parts)

    # reason: BenchmarkConfig exposes docs per sec/stub as its public contract; bundling would break callers.
    def to_result(  # ruff: ignore[too-many-arguments]
        self,
        *,
        docs_per_sec: float = 0.0,
        tokens_per_sec: float = 0.0,
        median_seconds: float = 0.0,
        min_seconds: float = 0.0,
        max_seconds: float = 0.0,
        peak_gpu_memory_mb: int = 0,
        exact_agree_f1: float = 0.0,
        containment_agree_f1: float = 0.0,
        exact_gold_f1: float = 0.0,
        containment_gold_f1: float = 0.0,
        stub: bool = False,
    ) -> BenchmarkResultRow:
        return BenchmarkResultRow(
            config_id=self.config_id,
            path_id=self.path_id,
            path_name=self.path_name,
            batch_size=self.batch_size,
            length_bucket=self.length_bucket,
            dtype=self.dtype,
            compile_mode=self.compile_mode,
            triton=self.triton,
            onnx_variant=self.onnx_variant,
            attention_kernel=self.attention_kernel,
            status="ok",
            docs_per_sec=docs_per_sec,
            tokens_per_sec=tokens_per_sec,
            median_seconds=median_seconds,
            min_seconds=min_seconds,
            max_seconds=max_seconds,
            peak_gpu_memory_mb=peak_gpu_memory_mb,
            exact_agree_f1=exact_agree_f1,
            containment_agree_f1=containment_agree_f1,
            exact_gold_f1=exact_gold_f1,
            containment_gold_f1=containment_gold_f1,
            stub=stub,
        )

    def to_skipped(self, *, reason: str | None = None, stub: bool = False) -> BenchmarkResultRow:
        skip_reason = reason or self.skip_reason
        if not skip_reason:
            msg = f"skip result requires a reason: {self.config_id}"
            raise ValueError(msg)
        return BenchmarkResultRow(
            config_id=self.config_id,
            path_id=self.path_id,
            path_name=self.path_name,
            batch_size=self.batch_size,
            length_bucket=self.length_bucket,
            dtype=self.dtype,
            compile_mode=self.compile_mode,
            triton=self.triton,
            onnx_variant=self.onnx_variant,
            attention_kernel=self.attention_kernel,
            status="skipped",
            skip_reason=skip_reason,
            stub=stub,
        )

    def to_error(self, error: str, *, stub: bool = False) -> BenchmarkResultRow:
        return BenchmarkResultRow(
            config_id=self.config_id,
            path_id=self.path_id,
            path_name=self.path_name,
            batch_size=self.batch_size,
            length_bucket=self.length_bucket,
            dtype=self.dtype,
            compile_mode=self.compile_mode,
            triton=self.triton,
            onnx_variant=self.onnx_variant,
            attention_kernel=self.attention_kernel,
            status="error",
            error=error,
            stub=stub,
        )


def build_config_registry() -> tuple[BenchmarkConfig, ...]:
    configs: list[BenchmarkConfig] = []
    configs.extend(_native_reference_configs())
    configs.extend(_native_batched_configs())
    configs.extend(_hf_pipeline_configs())
    configs.extend(_onnx_configs())
    return tuple(configs)


def _native_reference_configs() -> list[BenchmarkConfig]:
    return [
        BenchmarkConfig(
            path_id="A",
            path_name=PATH_NAMES["A"],
            batch_size=batch,
            length_bucket=bucket,
            dtype="native-bf16",
            compile_mode=compile_mode,
            triton=triton,
            onnx_variant="none",
            attention_kernel="native-banded",
            skip_reason=(None if batch == 1 else "native_reference_predict_text_is_batch1_only"),
        )
        for bucket in LENGTH_BUCKET_NAMES
        for batch in BATCH_SIZES
        for compile_mode in COMPILE_MODES
        for triton in TRITON_MODES
    ]


def _native_batched_configs() -> list[BenchmarkConfig]:
    return [
        BenchmarkConfig(
            path_id="A2",
            path_name=PATH_NAMES["A2"],
            batch_size=batch,
            length_bucket=bucket,
            dtype="native-bf16",
            compile_mode=compile_mode,
            triton=triton,
            onnx_variant="none",
            attention_kernel="native-banded",
        )
        for bucket in LENGTH_BUCKET_NAMES
        for batch in BATCH_SIZES
        for compile_mode in COMPILE_MODES
        for triton in TRITON_MODES
    ]


def _hf_pipeline_configs() -> list[BenchmarkConfig]:
    return [
        BenchmarkConfig(
            path_id="B",
            path_name=PATH_NAMES["B"],
            batch_size=batch,
            length_bucket=bucket,
            dtype=dtype,
            compile_mode=compile_mode,
            triton="na",
            onnx_variant="none",
            attention_kernel=attention_kernel,
        )
        for bucket in LENGTH_BUCKET_NAMES
        for batch in BATCH_SIZES
        for dtype in B_DTYPES
        for compile_mode in COMPILE_MODES
        for attention_kernel in ATTENTION_KERNELS
    ]


def _onnx_configs() -> list[BenchmarkConfig]:
    return [
        BenchmarkConfig(
            path_id="C",
            path_name=PATH_NAMES["C"],
            batch_size=batch,
            length_bucket=bucket,
            dtype=ONNX_DTYPE[variant],
            compile_mode="none",
            triton="na",
            onnx_variant=variant,
            attention_kernel="onnx-runtime",
        )
        for bucket in LENGTH_BUCKET_NAMES
        for batch in BATCH_SIZES
        for variant in ONNX_VARIANTS
    ]


def validate_registry_completeness(registry: tuple[BenchmarkConfig, ...]) -> list[str]:
    errors: list[str] = []
    _require_values(errors, "paths", {config.path_id for config in registry}, set(PATH_IDS))
    _require_values(
        errors,
        "batch_sizes",
        {config.batch_size for config in registry},
        set(BATCH_SIZES),
    )
    _require_values(
        errors,
        "length_buckets",
        {config.length_bucket for config in registry},
        set(LENGTH_BUCKET_NAMES),
    )

    by_path = {path_id: [config for config in registry if config.path_id == path_id] for path_id in PATH_IDS}
    for path_id, configs in by_path.items():
        if not configs:
            errors.append(f"missing path {path_id}")
            continue
        _require_values(
            errors,
            f"{path_id}.batch_sizes",
            {config.batch_size for config in configs},
            set(BATCH_SIZES),
        )
        _require_values(
            errors,
            f"{path_id}.length_buckets",
            {config.length_bucket for config in configs},
            set(LENGTH_BUCKET_NAMES),
        )

    for path_id in ("A", "A2", "B"):
        _require_values(
            errors,
            f"{path_id}.compile_modes",
            {config.compile_mode for config in by_path[path_id]},
            set(COMPILE_MODES),
        )
    for path_id in ("A", "A2"):
        _require_values(
            errors,
            f"{path_id}.triton_modes",
            {config.triton for config in by_path[path_id]},
            set(TRITON_MODES),
        )
    _require_values(errors, "B.dtypes", {config.dtype for config in by_path["B"]}, set(B_DTYPES))
    _require_values(
        errors,
        "B.attention_kernels",
        {config.attention_kernel for config in by_path["B"]},
        set(ATTENTION_KERNELS),
    )
    _require_values(
        errors,
        "C.onnx_variants",
        {config.onnx_variant for config in by_path["C"]},
        set(ONNX_VARIANTS),
    )

    missing_a_skip = [config.config_id for config in by_path["A"] if config.batch_size != 1 and not config.skip_reason]
    if missing_a_skip:
        errors.append(f"A batch>1 configs missing skip_reason: {missing_a_skip[:3]}")
    return errors


def select_representative_configs(
    registry: tuple[BenchmarkConfig, ...],
    *,
    batch_sizes: tuple[int, ...] = (1, 32, 128),
    bucket: LengthBucketName = "medium",
) -> tuple[BenchmarkConfig, ...]:
    """Cross-path sample for a cheap Stage-1 validation run.

    Every path on ONE length bucket across a few batch sizes, canonical on the
    secondary axes, excluding skipped configs (Stage-1 runs only real GPU
    configs). Answers "does each path load on GPU + which family wins" without
    the full ~627-config sweep. The full registry stays the 0-lazy source of
    truth; this only bounds the *run*.

    Returns:
        A subset covering every execution path once per requested batch size, on the one length
        bucket, canonical on the other axes, with skipped configs dropped. It bounds the run
        rather than the registry: the full sweep stays the source of truth, and nothing here
        removes a config from it.

    """
    chosen: list[BenchmarkConfig] = []
    for config in registry:
        if config.skip_reason or config.length_bucket != bucket:
            continue
        if config.batch_size not in batch_sizes:
            continue
        if config.path_id in {"A", "A2"} and not (config.compile_mode == "none" and config.triton == "on"):
            continue
        if config.path_id == "B" and not (
            config.compile_mode == "none" and config.dtype == "bf16" and config.attention_kernel == "default"
        ):
            continue
        if config.path_id == "C" and config.onnx_variant != "fp16":
            continue
        chosen.append(config)
    return tuple(chosen)


def _require_values(errors: list[str], name: str, actual: set[object], expected: set[object]) -> None:
    if actual != expected:
        missing = _display_values(expected - actual)
        extra = _display_values(actual - expected)
        errors.append(f"{name} mismatch: missing={missing} extra={extra}")


def _display_values(values: set[object]) -> list[str]:
    return sorted(str(value) for value in values)
