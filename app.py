#!/usr/bin/env python3
"""
CarePing OS v2.0 — Universal Application Entrypoint (CLI + Web Server + Streamlit)
Team Name: AAdharsha Kutumbam
Members  : Naga Adithya V (23BCE9062) & Harsha Vardhan Vurtha (23BCE9129)

Usage:
  Option 1 (Standalone Web App): Open `index.html` directly in any browser
  Option 2 (Python CLI + Server): python app.py
  Option 3 (Streamlit Mode)     : streamlit run app.py
"""

import os
import sys
import json
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_streamlit_mode():
    import streamlit as st
    import streamlit.components.v1 as components

    st.set_page_config(
        page_title="CarePing OS v2.0 — Queue & Retention Intelligence",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    html_path = os.path.join(BASE_DIR, "index.html")
    if not os.path.exists(html_path):
        html_path = os.path.join(BASE_DIR, "forgotten_patient_dashboard.html")
    with open(html_path, "r", encoding="utf-8") as f:
        html_code = f.read()
    components.html(html_code, height=1350, scrolling=True)


def main():
    if "streamlit" in sys.modules or any("streamlit" in arg for arg in sys.argv):
        try:
            run_streamlit_mode()
            return
        except Exception:
            pass

    metrics_path = os.path.join(BASE_DIR, "outputs", "model_metrics_and_weights.json")
    if not os.path.exists(metrics_path):
        import train_and_evaluate_model
        train_and_evaluate_model.main()

    with open(metrics_path, "r") as f:
        meta = json.load(f)

    print("=" * 78)
    print("⚡ CAREPING OS v2.0 — INTERACTIVE PROTOTYPE & ML TELEMETRY SERVER")
    print("=" * 78)
    print(f"  Team            : {meta['team']} ({', '.join(meta['members'])})")
    print(f"  ML Architecture : {meta['model_declaration']}")
    print(f"  Test R^2 / MAE  : R^2 = {meta['benchmark_80_20_split']['scratch_ols_closed_form']['r2']} | MAE = {meta['benchmark_80_20_split']['scratch_ols_closed_form']['mae_min']} min")
    print(f"  5-Fold CV R^2   : {meta['five_fold_cross_validation']['Scratch_OLS_ClosedForm']['cv_mean_r2']}")
    print(f"  Inference Speed : {meta['computational_efficiency']['inference_latency_microseconds_per_patient']} microseconds/patient")
    print("=" * 78)

    if "--no-server" in sys.argv:
        print("Verification complete (--no-server flag passed).")
        return

    port = int(os.environ.get("PORT", 8080))
    os.chdir(BASE_DIR)
    server = HTTPServer(("0.0.0.0", port), SimpleHTTPRequestHandler)
    url = f"http://localhost:{port}/index.html"
    print(f"\n🚀 Serving CarePing OS v2.0 Live Dashboard at: {url}")
    print("   (Press Ctrl+C to stop the server)\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down CarePing OS server.")


if __name__ == "__main__":
    main()
