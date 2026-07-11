"""Streamlit web UI mirroring `maid check`.

Run via `maid web` (which shells out to `streamlit run` on this file) or directly
with `streamlit run -m mac_ai_doctor.webapp`. Results reuse the same
`resolve_model` -> `estimate` path as the CLI, so the numbers and verdict policy
stay identical; only the presentation differs (native Streamlit widgets instead
of Rich renderables, which do not render in a browser).
"""

from __future__ import annotations

import streamlit as st

# Absolute imports: `streamlit run webapp.py` executes this file as a top-level
# script (__main__) with no package context, so relative imports would fail.
from mac_ai_doctor.cli import DISCLAIMER, VERDICT_GUIDANCE
from mac_ai_doctor.estimate import DTYPE_BYTES, estimate
from mac_ai_doctor.metadata import resolve_model
from mac_ai_doctor.models import Estimate, Verdict
from mac_ai_doctor.sharing import badge_svg, build_share_query, markdown_report
from mac_ai_doctor.system import detect_system

_VERDICT_STYLE = {
    Verdict.COMFORTABLE: ("🟢", st.success),
    Verdict.TIGHT: ("🟡", st.warning),
    Verdict.UNLIKELY: ("🔴", st.error),
    Verdict.UNKNOWN: ("⚪", st.info),
}


def _fmt_gb(value: float | None) -> str:
    return f"{value:.2f} GB" if value is not None else "unknown"


def _render_result(result: Estimate) -> None:
    st.subheader(result.model.model_id)

    icon, banner = _VERDICT_STYLE[result.verdict]
    banner(
        f"{icon} **{result.verdict.value}** · {result.confidence} confidence — "
        f"{VERDICT_GUIDANCE[result.verdict]}"
    )

    peak = (
        f"{result.low_gb:.2f}–{result.high_gb:.2f} GB"
        if result.low_gb is not None and result.high_gb is not None
        else "unknown"
    )
    left, right = st.columns(2)
    left.metric("Peak range", peak)
    right.metric("Available", f"{result.available_gb:.1f} GB")

    st.table(
        {
            "Component": ["Weights", "KV cache", "Runtime"],
            "Estimate": [
                _fmt_gb(result.weights_gb),
                _fmt_gb(result.kv_cache_gb),
                _fmt_gb(result.runtime_gb),
            ],
        }
    )

    if result.assumptions:
        with st.expander("Assumptions"):
            for note in result.assumptions:
                st.markdown(f"- {note}")

    st.caption(DISCLAIMER)

    st.divider()
    st.markdown("#### Share this check")
    query = build_share_query(
        result.model.model_id,
        result.available_gb,
        result.context,
        result.concurrency,
        result.kv_dtype,
    )
    st.caption("Copy this query onto the end of the app URL to reproduce the same inputs.")
    st.code(f"?{query}", language=None)
    report_col, badge_col = st.columns(2)
    report_col.download_button(
        "Download Markdown report",
        markdown_report(result),
        file_name="mac-ai-doctor-report.md",
        mime="text/markdown",
        use_container_width=True,
    )
    badge_col.download_button(
        "Download SVG badge",
        badge_svg("Mac memory fit", result.verdict),
        file_name="mac-ai-doctor-badge.svg",
        mime="image/svg+xml",
        use_container_width=True,
    )


def main() -> None:
    st.set_page_config(page_title="Mac AI Doctor", page_icon="🩺", layout="centered")
    st.title("🩺 Mac AI Doctor")
    st.markdown("### Will this exact model fit your Mac?")
    st.caption(
        "Paste a Hugging Face URL or model ID. Get a transparent unified-memory estimate "
        "before downloading the weights."
    )

    detected = detect_system()
    params = st.query_params
    memory_default = float(params.get("memory", detected.memory_gb or 16.0))
    model_default = str(params.get("model", ""))
    context_default = int(params.get("context", 4096))
    concurrency_default = int(params.get("concurrency", 1))
    dtype_default = str(params.get("kv_dtype", "fp16"))
    dtypes = list(DTYPE_BYTES)
    if dtype_default not in dtypes:
        dtype_default = "fp16"

    st.info("Only small metadata files are read. Model weights are never downloaded.")
    with st.form("check"):
        model = st.text_input(
            "Hugging Face model, URL, or local file",
            value=model_default,
            placeholder="https://huggingface.co/mlx-community/Qwen3-8B-4bit",
            help="Also accepts a local .gguf file or directory containing config.json.",
        )
        col1, col2 = st.columns(2)
        memory_gb = col1.number_input(
            "Unified memory (GB)", min_value=0.5, value=memory_default, step=1.0
        )
        context = col2.number_input("Context length", min_value=1, value=context_default, step=512)
        col3, col4 = st.columns(2)
        concurrency = col3.number_input(
            "Concurrent requests", min_value=1, value=concurrency_default, step=1
        )
        kv_dtype = col4.selectbox("KV cache dtype", dtypes, index=dtypes.index(dtype_default))
        submitted = st.form_submit_button("Check fit", type="primary", use_container_width=True)

    if not submitted:
        return
    if not model.strip():
        st.error("Enter a model reference (HF ID, GGUF path, or local config dir).")
        return

    with st.spinner(f"Resolving {model.strip()}…"):
        try:
            info = resolve_model(model.strip())
            result = estimate(info, float(memory_gb), int(context), int(concurrency), kv_dtype)
        except Exception as exc:  # noqa: BLE001 — surface any resolve/estimate failure inline
            st.error(f"Could not read model metadata: {exc}")
            return

    _render_result(result)


if __name__ == "__main__":
    main()
