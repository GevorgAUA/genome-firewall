"""Genome Firewall Streamlit interface."""

from __future__ import annotations

import html
import json
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Any

import pandas as pd
import streamlit as st

from genome_firewall.config import load_config
from genome_firewall.predict import WARNING, predict_cohort_sample, predict_genome

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "mvp.yaml"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ANALYSIS_LOCK = Lock()
EPHEMERAL_INFERENCE = os.environ.get(
    "GENOME_FIREWALL_EPHEMERAL_INFERENCE",
    "0",
).strip().casefold() in {"1", "true", "yes"}
INFERENCE_CACHE_ROOT = PROJECT_ROOT / "data" / "amrfinder" / "inference"

DEMO_SAMPLES = (
    {
        "sample_id": "562.100008",
        "title": "Mixed response",
        "description": "Ciprofloxacin likely to work · Ampicillin likely to fail",
    },
    {
        "sample_id": "562.45637",
        "title": "Safety no-call",
        "description": "Ciprofloxacin likely to fail · Ampicillin no-call",
    },
    {
        "sample_id": "562.100190",
        "title": "Both likely to work",
        "description": "Confident likely-to-work calls for both antibiotics",
    },
)

DECISION_STYLE = {
    "likely_to_work": ("LIKELY TO WORK", "work", "✓"),
    "likely_to_fail": ("LIKELY TO FAIL", "fail", "!"),
    "no_call": ("NO CALL", "uncertain", "?"),
}


def _page_styles() -> None:
    st.html(
        """
        <style>
        .stApp { background: #f7f8fa; color: #132238; }
        .block-container { max-width: 1180px; padding-top: 2.2rem; padding-bottom: 4rem; }
        .gf-hero {
            background: #10233f;
            border: 1px solid #203a5f;
            border-radius: 22px;
            color: white;
            padding: 2.15rem 2.35rem 2rem;
            box-shadow: 0 18px 45px rgba(15, 35, 63, 0.14);
            margin-bottom: 1.1rem;
        }
        .gf-eyebrow { color: #76e0c2; font-size: .78rem; font-weight: 750; letter-spacing: .13em; }
        .gf-hero h1 { color: white; font-size: clamp(2rem, 5vw, 3.4rem); margin: .35rem 0 .45rem; }
        .gf-hero p { color: #d7e2ef; font-size: 1.08rem; max-width: 760px; margin: 0 0 1.2rem; }
        .gf-pill {
            display: inline-block; background: #183454; border: 1px solid #315476;
            border-radius: 999px; color: #eaf2f8; font-size: .78rem; font-weight: 650;
            margin: .2rem .35rem .1rem 0; padding: .36rem .68rem;
        }
        .gf-panel {
            background: white; border: 1px solid #e2e8f0; border-radius: 18px;
            padding: 1.25rem 1.3rem .8rem; box-shadow: 0 8px 25px rgba(15, 35, 63, .05);
        }
        .gf-section-title { font-size: 1.1rem; font-weight: 750; margin-bottom: .15rem; }
        .gf-muted { color: #64748b; font-size: .9rem; }
        .gf-result-header {
            display: flex; align-items: center; justify-content: space-between; gap: 1rem;
            background: white; border: 1px solid #e2e8f0; border-radius: 15px;
            padding: .85rem 1rem; margin: .3rem 0 .8rem;
        }
        .gf-result-name { font-size: 1.15rem; font-weight: 780; }
        .gf-badge {
            border-radius: 999px; font-size: .76rem; font-weight: 800;
            padding: .45rem .68rem;
        }
        .gf-badge.work { background: #d9f7eb; color: #087252; }
        .gf-badge.fail { background: #fee2e2; color: #b42318; }
        .gf-badge.uncertain { background: #fff0c2; color: #8a5800; }
        .gf-step {
            background: white; border: 1px solid #e2e8f0; border-radius: 14px;
            min-height: 108px; padding: .9rem 1rem;
        }
        .gf-step strong { display: block; color: #16385f; margin-bottom: .25rem; }
        .gf-step span { color: #64748b; font-size: .86rem; }
        div.stButton > button[kind="primary"] { font-weight: 750; min-height: 3rem; }
        [data-testid="stFileUploaderDropzone"] { background: #f8fafc; border-radius: 14px; }
        @media (max-width: 700px) {
            .block-container { padding-top: 1rem; }
            .gf-hero { padding: 1.5rem 1.25rem; border-radius: 17px; }
            .gf-result-header { align-items: flex-start; flex-direction: column; }
        }
        </style>
        """
    )


def _hero() -> None:
    st.html(
        """
        <section class="gf-hero">
          <div class="gf-eyebrow">INTERPRETABLE AMR RESEARCH PROTOTYPE</div>
          <h1>Genome Firewall</h1>
          <p>Turn an assembled <em>E. coli</em> genome into calibrated antibiotic-resistance
          predictions, explicit uncertainty, confirmed drug targets, and inspectable evidence.</p>
          <span class="gf-pill">E. coli</span>
          <span class="gf-pill">Ciprofloxacin + Ampicillin</span>
          <span class="gf-pill">Target-gated</span>
          <span class="gf-pill">Evidence-rich</span>
        </section>
        """
    )
    st.warning(f"⚠️ {WARNING}")


def _run_uploaded(uploaded: Any, antibiotic: str | None) -> list[dict[str, Any]]:
    suffix = Path(uploaded.name).suffix.lower()
    if suffix not in {".fa", ".fasta", ".fna", ".fas"}:
        suffix = ".fna"
    with TemporaryDirectory(prefix="genome_firewall_") as directory:
        fasta_path = Path(directory) / f"uploaded{suffix}"
        fasta_path.write_bytes(uploaded.getvalue())
        # AMRFinder is CPU-heavy. Serialize hosted jobs to avoid exhausting the
        # small Space and to prevent duplicate uploads racing on cached outputs.
        with ANALYSIS_LOCK:
            try:
                return predict_genome(
                    fasta_path,
                    antibiotic,
                    config_path=CONFIG_PATH,
                    timeout_seconds=300,
                )
            finally:
                if EPHEMERAL_INFERENCE:
                    shutil.rmtree(INFERENCE_CACHE_ROOT, ignore_errors=True)


def _summary_table(results: list[dict[str, Any]]) -> None:
    rows = []
    for result in results:
        label, _, _ = DECISION_STYLE[result["prediction"]]
        probability = result["p_resistant"]
        rows.append(
            {
                "Antibiotic": result["display_name"],
                "Prediction": label.title(),
                "Resistance probability": (
                    f"{probability:.1%}" if probability is not None else "Unavailable"
                ),
                "Confidence": (
                    f"{result['confidence']:.1%}"
                    if result["confidence"] is not None
                    else "Unavailable"
                ),
                "Target": result["target_status"].title(),
                "Evidence": result["evidence_category"].replace("_", " ").title(),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _render_result(result: dict[str, Any]) -> None:
    label, css_class, icon = DECISION_STYLE[result["prediction"]]
    name = html.escape(result["display_name"])
    st.html(
        f"""
        <div class="gf-result-header">
          <div><span class="gf-result-name">{name}</span><br>
          <span class="gf-muted">Probability of resistant phenotype</span></div>
          <span class="gf-badge {css_class}">{icon} {label}</span>
        </div>
        """
    )
    probability = result["p_resistant"]
    if probability is None:
        st.error(f"Prediction unavailable: {result['no_call_reason'].replace('_', ' ')}")
        return

    metric_columns = st.columns(3)
    metric_columns[0].metric("Resistance probability", f"{probability:.1%}")
    metric_columns[1].metric("Decision confidence", f"{result['confidence']:.1%}")
    metric_columns[2].metric("Target status", result["target_status"].title())
    st.progress(
        round(probability * 100),
        text=f"P(resistant) = {probability:.3f}",
    )
    if result["no_call_reason"]:
        st.info(f"No-call reason: {result['no_call_reason'].replace('_', ' ')}")
    target_genes = ", ".join(result["target_genes_found"]) or "None confirmed"
    st.caption(
        f"Target evidence: {target_genes} · Source: "
        f"{(result['target_evidence_source'] or 'none').replace('_', ' ')}"
    )

    with st.expander(f"Known resistance evidence ({len(result['evidence'])})"):
        if result["evidence"]:
            evidence_frame = pd.DataFrame(result["evidence"])
            preferred = [
                "element_symbol",
                "element_name",
                "class",
                "subclass",
                "method",
                "coverage",
                "identity",
            ]
            st.dataframe(
                evidence_frame[[column for column in preferred if column in evidence_frame]],
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.write("No directly mapped AMRFinder resistance determinant was detected.")

    with st.expander("Statistical feature influences"):
        st.caption(
            "Coefficients describe model associations in this dataset; they do not prove "
            "biological causation."
        )
        if result["model_feature_contributions"]:
            st.dataframe(
                pd.DataFrame(result["model_feature_contributions"]),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.write("No detected feature was present in the frozen training vocabulary.")

    if result["unseen_elements"]:
        with st.expander(f"Unseen AMR elements ({len(result['unseen_elements'])})"):
            st.write(", ".join(result["unseen_elements"]))


def _results(results: list[dict[str, Any]]) -> None:
    st.divider()
    st.subheader("Prediction report")
    st.caption(f"Sample: {results[0]['sample_id']}")
    _summary_table(results)
    tabs = st.tabs([result["display_name"] for result in results])
    for tab, result in zip(tabs, results, strict=True):
        with tab:
            _render_result(result)
    st.download_button(
        "Download JSON report",
        data=json.dumps(results, indent=2, sort_keys=True),
        file_name=f"genome_firewall_{results[0]['sample_id']}.json",
        mime="application/json",
        use_container_width=True,
    )
    st.warning(f"⚠️ {WARNING}")


def _workflow() -> None:
    st.subheader("What happens to your genome")
    columns = st.columns(4)
    steps = [
        ("01 · Validate", "Checks FASTA structure before processing."),
        ("02 · Annotate", "AMRFinder identifies genes and mutations."),
        ("03 · Predict", "Frozen calibrated models estimate resistance."),
        ("04 · Safeguard", "Target gates and uncertainty prevent unsafe calls."),
    ]
    for column, (title, description) in zip(columns, steps, strict=True):
        with column:
            st.html(
                f'<div class="gf-step"><strong>{title}</strong>'
                f"<span>{description}</span></div>"
            )


def _validation_snapshot() -> None:
    metrics_path = PROJECT_ROOT / "reports" / "metrics.csv"
    if not metrics_path.is_file():
        return
    metrics = pd.read_csv(metrics_path)
    rows = []
    for row in metrics.to_dict(orient="records"):
        rows.append(
            {
                "Antibiotic": str(row["antibiotic"]).title(),
                "Test genomes": int(row["test_samples"]),
                "Balanced accuracy": f"{float(row['balanced_accuracy']):.1%}",
                "AUROC": f"{float(row['auroc']):.1%}",
                "Called accuracy": f"{float(row['called_accuracy']):.1%}",
                "No-call rate": f"{float(row['no_call_rate']):.1%}",
            }
        )
    with st.expander("Blind holdout performance · frozen before interface development"):
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption(
            "Twenty-four test genomes per antibiotic. This small research holdout supports "
            "the demo but is not clinical validation."
        )


def main() -> None:
    st.set_page_config(
        page_title="Genome Firewall · AMR Prediction",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _page_styles()
    _hero()
    config = load_config(CONFIG_PATH)

    left, right = st.columns([1.65, 1], gap="large")
    with left:
        st.html('<div class="gf-section-title">Analyze an assembled genome</div>')
        st.caption("Upload a nucleotide FASTA for the configured E. coli workflow (maximum 25 MB).")
        uploaded = st.file_uploader(
            "Genome FASTA",
            type=["fa", "fasta", "fna", "fas"],
            help="Assembled nucleotide FASTA only; raw sequencing reads are not supported.",
        )
        options = {
            "All supported antibiotics": None,
            "Ciprofloxacin": "ciprofloxacin",
            "Ampicillin": "ampicillin",
        }
        selected_label = st.selectbox("Antibiotic", options)
        selected_antibiotic = options[selected_label]
        if uploaded is not None and uploaded.size > MAX_UPLOAD_BYTES:
            st.error("This file exceeds the 25 MB MVP safety limit.")
        run_disabled = uploaded is None or uploaded.size > MAX_UPLOAD_BYTES
        if st.button(
            "Run prediction",
            type="primary",
            use_container_width=True,
            disabled=run_disabled,
        ):
            with st.spinner("Running AMRFinder and calibrated models…"):
                st.session_state["prediction_results"] = _run_uploaded(
                    uploaded,
                    selected_antibiotic,
                )
    with right:
        st.html('<div class="gf-section-title">Instant demo gallery</div>')
        st.caption(
            "Three precomputed genomes showcase different decisions immediately. "
            "Demo buttons always display both antibiotics."
        )
        for index, demo in enumerate(DEMO_SAMPLES, start=1):
            st.markdown(f"**{index}. {demo['title']}** · `{demo['sample_id']}`")
            st.caption(demo["description"])
            if st.button(
                f"Run demo {index} · {demo['title']}",
                key=f"demo_{demo['sample_id']}",
                use_container_width=True,
            ):
                with st.spinner("Loading frozen demo artifacts…"):
                    st.session_state["prediction_results"] = predict_cohort_sample(
                        demo["sample_id"],
                        config,
                        None,
                    )
        if EPHEMERAL_INFERENCE:
            st.info(
                "A new FASTA normally takes tens of seconds. Hosted uploads and derived "
                "reports are deleted immediately after prediction."
            )
        else:
            st.info(
                "A new FASTA normally takes tens of seconds on this machine; repeat files use "
                "cached annotation."
            )

    results = st.session_state.get("prediction_results")
    if results:
        _results(results)
    st.divider()
    _validation_snapshot()
    _workflow()
    st.caption("Genome Firewall MVP · E. coli only · Model artifacts frozen after blind evaluation")


if __name__ == "__main__":
    main()
