from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from meddies_pii.pdf_redaction.errors import VerificationError
from meddies_pii.pdf_redaction.verification_scans import (
    _first_failed_run,
    _has_json_attachments,
    _matched_hashes,
    _matched_string_hashes,
    _observation,
    _output_digest,
    _present_risky_markers,
    _qpdf_json_string_values,
    _tool_failure_finding,
)
from meddies_pii.pdf_redaction.verification_types import (
    SubprocessToolRunner,
    ToolObservation,
    ToolRunner,
    VerificationFinding,
    VerificationReport,
    VerificationToolchain,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from meddies_pii.pdf_redaction.writers import RedactionWriteResult


class IndependentPdfVerifier:
    def __init__(
        self,
        *,
        toolchain: VerificationToolchain | None = None,
        runner: ToolRunner | None = None,
    ) -> None:
        self._toolchain = toolchain or VerificationToolchain()
        self._runner = runner or SubprocessToolRunner()

    # reason: Structure, object, text, freshness, and digest checks produce one independent verification report.
    def verify(  # ruff: ignore[too-many-locals]
        self,
        written: RedactionWriteResult,
        *,
        known_canaries: Sequence[str],
    ) -> VerificationReport:
        if not known_canaries:
            msg = "verification requires at least one known synthetic value"
            raise VerificationError(
                msg,
                stage="verify",
                code="empty_canary_set",
            )
        observations: list[ToolObservation] = []
        findings = [
            self._check_writer_safety(written),
            self._check_fresh_output(written),
        ]

        structure_finding, structure_observation = self._check_structure(written)
        findings.append(structure_finding)
        observations.append(structure_observation)

        expanded_finding, expanded_observation = self._check_expanded_objects(
            written,
            known_canaries,
        )
        findings.append(expanded_finding)
        observations.append(expanded_observation)

        risky_finding, risky_observations = self._check_risky_objects(
            written,
            known_canaries,
        )
        findings.append(risky_finding)
        observations.extend(risky_observations)

        text_finding, text_observations = self._check_text(written, known_canaries)
        findings.append(text_finding)
        observations.extend(text_observations)

        with TemporaryDirectory(prefix="meddies-pdf-verify-") as directory:
            render_finding, render_observation, rendered_pages = self._check_render(
                written,
                Path(directory),
            )
            findings.append(render_finding)
            observations.append(render_observation)
            ocr_finding, ocr_observations = self._check_ocr(
                rendered_pages,
                known_canaries,
            )
            findings.append(ocr_finding)
            observations.extend(ocr_observations)

        output_sha256 = _output_digest(written.output_path)
        findings[1] = self._check_fresh_output(
            written,
            output_sha256=output_sha256,
        )
        return VerificationReport(
            output_path=written.output_path,
            output_sha256=output_sha256,
            passed=all(finding.passed for finding in findings),
            findings=tuple(findings),
            observations=tuple(observations),
        )

    @staticmethod
    def _check_writer_safety(
        written: RedactionWriteResult,
    ) -> VerificationFinding:
        destructive = written.safety == "destructive"
        return VerificationFinding(
            gate="writer_safety",
            passed=destructive,
            code="destructive_writer" if destructive else "unsafe_writer",
            detail=(
                "writer declares a destructive output mechanism"
                if destructive
                else "overlay-only output is a calibration control and cannot pass"
            ),
        )

    @staticmethod
    def _check_fresh_output(
        written: RedactionWriteResult,
        *,
        output_sha256: str | None = None,
    ) -> VerificationFinding:
        output = written.output_path
        source = written.source_path
        observed_sha256 = _output_digest(output) if output_sha256 is None else output_sha256
        try:
            fresh = (
                output.is_file()
                and source.is_file()
                and output.resolve() != source.resolve()
                and output.stat().st_ino != source.stat().st_ino
                and observed_sha256 == written.output_sha256
            )
        except OSError:
            fresh = False
        return VerificationFinding(
            gate="fresh_output",
            passed=fresh,
            code="fresh_output" if fresh else "output_identity_mismatch",
            detail=(
                "output is a fresh file with the recorded digest"
                if fresh
                else "output is missing, aliases its source, or has changed since writing"
            ),
        )

    def _check_structure(
        self,
        written: RedactionWriteResult,
    ) -> tuple[VerificationFinding, ToolObservation]:
        run = self._runner.run(
            (self._toolchain.qpdf, "--check", str(written.output_path)),
            timeout_seconds=self._toolchain.timeout_seconds,
        )
        observation = _observation("structure", run)
        if run.status != "completed" or run.returncode != 0:
            return _tool_failure_finding("structure", run), observation
        return (
            VerificationFinding(
                gate="structure",
                passed=True,
                code="structure_valid",
                detail="qpdf accepted the output structure",
            ),
            observation,
        )

    def _check_expanded_objects(
        self,
        written: RedactionWriteResult,
        known_canaries: Sequence[str],
    ) -> tuple[VerificationFinding, ToolObservation]:
        with TemporaryDirectory(prefix="meddies-pdf-qdf-") as directory:
            expanded = Path(directory) / "expanded.pdf"
            run = self._runner.run(
                (
                    self._toolchain.qpdf,
                    "--qdf",
                    "--object-streams=disable",
                    "--decode-level=generalized",
                    str(written.output_path),
                    str(expanded),
                ),
                timeout_seconds=self._toolchain.timeout_seconds,
            )
            observation = _observation("expanded_objects", run)
            if run.status != "completed" or run.returncode != 0:
                return _tool_failure_finding("expanded_objects", run), observation
            if not expanded.is_file():
                return (
                    VerificationFinding(
                        gate="expanded_objects",
                        passed=False,
                        code="expanded_output_missing",
                        detail="qpdf returned success without an expanded object file",
                    ),
                    observation,
                )
            matched = _matched_hashes(expanded.read_bytes(), known_canaries)
            return (
                VerificationFinding(
                    gate="expanded_objects",
                    passed=not matched,
                    code=("expanded_objects_clean" if not matched else "known_text_in_expanded_objects"),
                    detail=(
                        "known values were absent from qpdf-expanded objects"
                        if not matched
                        else "one or more known values remained in expanded PDF objects"
                    ),
                    matched_value_sha256=matched,
                ),
                observation,
            )

    def _check_risky_objects(
        self,
        written: RedactionWriteResult,
        known_canaries: Sequence[str],
    ) -> tuple[VerificationFinding, tuple[ToolObservation, ...]]:
        run = self._runner.run(
            (self._toolchain.qpdf, "--json", str(written.output_path)),
            timeout_seconds=self._toolchain.timeout_seconds,
        )
        observations = (_observation("risky_objects", run),)
        if run.status != "completed" or run.returncode != 0:
            return _tool_failure_finding("risky_objects", run), observations

        json_output = run.stdout
        try:
            inventory = json.loads(json_output)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return (
                VerificationFinding(
                    gate="risky_objects",
                    passed=False,
                    code="object_inventory_invalid_json",
                    detail="qpdf returned an invalid JSON object inventory",
                ),
                observations,
            )
        if not isinstance(inventory, dict):
            return (
                VerificationFinding(
                    gate="risky_objects",
                    passed=False,
                    code="object_inventory_invalid_schema",
                    detail="qpdf returned a non-object JSON inventory",
                ),
                observations,
            )

        markers = _present_risky_markers(json_output)
        matched = _matched_string_hashes(
            _qpdf_json_string_values(inventory),
            known_canaries,
        )
        attachments_present = _has_json_attachments(inventory)
        passed = not attachments_present and not markers and not matched
        if matched:
            code = "known_text_in_object_inventory"
            detail = "one or more known values remained in the qpdf object inventory"
        elif passed:
            code = "risk_inventory_clean"
            detail = "qpdf found no attachments or blocked object markers"
        else:
            code = "risky_objects_present"
            detail = "qpdf found attachments or blocked object markers"
        return (
            VerificationFinding(
                gate="risky_objects",
                passed=passed,
                code=code,
                detail=detail,
                matched_value_sha256=matched,
                risky_markers=markers,
            ),
            observations,
        )

    def _check_text(
        self,
        written: RedactionWriteResult,
        known_canaries: Sequence[str],
    ) -> tuple[VerificationFinding, tuple[ToolObservation, ...]]:
        runs = tuple(
            self._runner.run(
                (
                    self._toolchain.pdftotext,
                    mode,
                    str(written.output_path),
                    "-",
                ),
                timeout_seconds=self._toolchain.timeout_seconds,
            )
            for mode in ("-raw", "-layout")
        )
        observations = tuple(_observation("text_extraction", run) for run in runs)
        failed_run = _first_failed_run(runs)
        if failed_run is not None:
            return _tool_failure_finding("text_extraction", failed_run), observations

        extracted = b"\n".join(run.stdout for run in runs)
        matched = _matched_hashes(extracted, known_canaries)
        return (
            VerificationFinding(
                gate="text_extraction",
                passed=not matched,
                code="text_absent" if not matched else "known_text_recoverable",
                detail=(
                    "known values were absent from independent text extraction"
                    if not matched
                    else "one or more known values remained independently extractable"
                ),
                matched_value_sha256=matched,
            ),
            observations,
        )

    def _check_render(
        self,
        written: RedactionWriteResult,
        directory: Path,
    ) -> tuple[VerificationFinding, ToolObservation, tuple[Path, ...]]:
        prefix = directory / "page"
        run = self._runner.run(
            (
                self._toolchain.pdftoppm,
                "-png",
                "-r",
                str(self._toolchain.render_dpi),
                str(written.output_path),
                str(prefix),
            ),
            timeout_seconds=self._toolchain.timeout_seconds,
        )
        observation = _observation("reopen_render", run)
        if run.status != "completed" or run.returncode != 0:
            return _tool_failure_finding("reopen_render", run), observation, ()
        rendered_pages = tuple(sorted(directory.glob("page-*.png")))
        passed = len(rendered_pages) == written.page_count
        return (
            VerificationFinding(
                gate="reopen_render",
                passed=passed,
                code="rendered" if passed else "render_page_count_mismatch",
                detail=(
                    "Poppler reopened and rendered every output page"
                    if passed
                    else "Poppler-rendered page count did not match the writer result"
                ),
            ),
            observation,
            rendered_pages,
        )

    def _check_ocr(
        self,
        rendered_pages: Sequence[Path],
        known_canaries: Sequence[str],
    ) -> tuple[VerificationFinding, tuple[ToolObservation, ...]]:
        if not rendered_pages:
            return (
                VerificationFinding(
                    gate="ocr",
                    passed=False,
                    code="render_prerequisite_failed",
                    detail="independent OCR could not run without rendered pages",
                ),
                (),
            )
        runs = tuple(
            self._runner.run(
                (
                    self._toolchain.tesseract,
                    str(page),
                    "stdout",
                    "-l",
                    self._toolchain.ocr_languages,
                    "--psm",
                    "6",
                ),
                timeout_seconds=self._toolchain.timeout_seconds,
            )
            for page in rendered_pages
        )
        observations = tuple(_observation("ocr", run) for run in runs)
        failed_run = _first_failed_run(runs)
        if failed_run is not None:
            return _tool_failure_finding("ocr", failed_run), observations

        matched = _matched_hashes(b"\n".join(run.stdout for run in runs), known_canaries)
        return (
            VerificationFinding(
                gate="ocr",
                passed=not matched,
                code="ocr_clean" if not matched else "known_text_visible_to_ocr",
                detail=(
                    "known values were absent from independent OCR"
                    if not matched
                    else "one or more known values remained visible to independent OCR"
                ),
                matched_value_sha256=matched,
            ),
            observations,
        )
