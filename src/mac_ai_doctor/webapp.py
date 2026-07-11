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


def main() -> None:
    st.set_page_config(page_title="Mac AI Doctor", page_icon="🩺")
    st.title("🩺 Mac AI Doctor")
    st.caption("Will this model fit in your unified memory? Weights are never downloaded.")

    detected = detect_system()
    memory_default = float(detected.memory_gb) if detected.memory_gb else 16.0

    with st.form("check"):
        model = st.text_input(
            "Model reference",
            placeholder="ibm-granite/granite-3.1-8b-instruct",
            help="Hugging Face ID, local .gguf file, or a directory with config.json.",
        )
        col1, col2 = st.columns(2)
        memory_gb = col1.number_input("Memory (GB)", min_value=0.5, value=memory_default, step=1.0)
        context = col2.number_input("Context", min_value=1, value=4096, step=512)
        col3, col4 = st.columns(2)
        concurrency = col3.number_input("Concurrency", min_value=1, value=1, step=1)
        dtypes = list(DTYPE_BYTES)
        kv_dtype = col4.selectbox("KV dtype", dtypes, index=dtypes.index("fp16"))
        submitted = st.form_submit_button("Check", type="primary")

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
