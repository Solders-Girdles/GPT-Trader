"""
Per-run reporting utilities: create a compact HTML report with key visuals.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def _save_plot_to_png_bytes() -> bytes:
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close()
    buf.seek(0)
    return buf.read()


def _img_tag_from_png_bytes(png_bytes: bytes, alt: str) -> str:
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return (
        f'<img alt="{alt}" src="data:image/png;base64,{b64}" style="max-width:100%;height:auto;"/>'
    )


def _flatten_params(param_obj: dict[str, Any]) -> dict[str, Any]:
    flat = {}
    for k, v in (param_obj or {}).items():
        flat[f"param_{k}"] = v
    return flat


def _build_topk_df(results: list[dict[str, Any]], k: int = 20) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for r in sorted(results, key=lambda x: x.get("sharpe", float("-inf")), reverse=True)[:k]:
        row = {
            "sharpe": r.get("sharpe", None),
            "cagr": r.get("cagr", None),
            "max_drawdown": r.get("max_drawdown", None),
            "n_trades": r.get("n_trades", None),
            "n_symbols": r.get("n_symbols", None),
        }
        row.update(_flatten_params(r.get("params", {})))
        rows.append(row)
    return pd.DataFrame(rows)


def _write_evo_progress_csv(output_dir: Path, evo_history: list[dict[str, Any]]) -> Path | None:
    if not evo_history:
        return None
    df = pd.DataFrame(evo_history)
    out_path = output_dir / "evo_progress.csv"
    df.to_csv(out_path, index=False)
    return out_path


def create_run_report(
    output_dir: Path,
    results: list[dict[str, Any]],
    evo_history: list[dict[str, Any]] | None,
    summary: dict[str, Any] | None,
) -> Path:
    """Generate a compact HTML report for a single run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Top-K table
    topk_df = _build_topk_df(results, k=20)
    topk_path = output_dir / "topk.csv"
    if not topk_df.empty:
        topk_df.to_csv(topk_path, index=False)

    # Plot: Pareto Sharpe vs MaxDD
    pareto_img = ""
    if not topk_df.empty:
        plt.figure(figsize=(6, 4))
        x = topk_df["max_drawdown"].astype(float)
        y = topk_df["sharpe"].astype(float)
        plt.scatter(x, y, c="tab:blue", alpha=0.7, edgecolors="none")
        plt.xlabel("Max Drawdown")
        plt.ylabel("Sharpe")
        plt.title("Pareto: Sharpe vs Max Drawdown (Top-20)")
        pareto_img = _img_tag_from_png_bytes(_save_plot_to_png_bytes(), "Pareto")

    # Plot: Evolution best/avg per generation
    evo_img = ""
    if evo_history:
        df = pd.DataFrame(evo_history)
        # Try common key names
        best_key = next((k for k in df.columns if k in ("best_fitness", "best_sharpe")), None)
        avg_key = next((k for k in df.columns if k in ("avg_fitness", "avg_sharpe")), None)
        gen_key = next((k for k in df.columns if k in ("generation",)), None)
        if best_key and gen_key:
            plt.figure(figsize=(6, 4))
            plt.plot(df[gen_key], df[best_key], label="Best")
            if avg_key:
                plt.plot(df[gen_key], df[avg_key], label="Avg")
            plt.xlabel("Generation")
            plt.ylabel("Sharpe")
            plt.title("Evolution Progress")
            plt.legend()
            evo_img = _img_tag_from_png_bytes(_save_plot_to_png_bytes(), "Evolution Progress")

    # Quick summary table
    summary_html = ""
    if summary:
        cfg = summary.get("config", {})
        summary.get("statistics", {})
        best = summary.get("best_result") or {}
        total = summary.get("total_evaluations", 0)
        summary_html = f"""
            <table border=0 cellpadding=6>
              <tr><td><b>Name</b></td><td>{cfg.get('name','')}</td></tr>
              <tr><td><b>Strategy</b></td><td>{cfg.get('strategy','')}</td></tr>
              <tr><td><b>Symbols</b></td><td>{', '.join(cfg.get('symbols', []))}</td></tr>
              <tr><td><b>Date Range</b></td><td>{cfg.get('date_range','')}</td></tr>
              <tr><td><b>Total Evaluations</b></td><td>{total}</td></tr>
              <tr><td><b>Best Sharpe</b></td><td>{best.get('sharpe','')}</td></tr>
              <tr><td><b>Best MaxDD</b></td><td>{best.get('max_drawdown','')}</td></tr>
            </table>
        """

    # HTML assembly
    report_html = f"""
    <html>
      <head>
        <meta charset="utf-8"/>
        <title>Optimization Report</title>
        <style>
          body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; padding: 16px; }}
          h2 {{ margin-top: 1.5em; }}
          table.dataframe {{ border-collapse: collapse; width: 100%; }}
          table.dataframe th, table.dataframe td {{ border: 1px solid #ddd; padding: 6px; font-size: 12px; }}
          table.dataframe th {{ background: #f5f5f5; }}
        </style>
      </head>
      <body>
        <h1>Optimization Report</h1>
        {summary_html}
        <h2>Pareto (Top-20)</h2>
        {pareto_img or '<i>No results available</i>'}
        <h2>Evolution Progress</h2>
        {evo_img or '<i>No evolutionary history</i>'}
        <h2>Top-20 Table</h2>
        {topk_df.head(20).to_html(index=False) if not topk_df.empty else '<i>No results</i>'}
        <p><small>Files: topk.csv{', evo_progress.csv' if evo_history else ''}</small></p>
      </body>
    </html>
    """

    report_path = output_dir / "report.html"
    with open(report_path, "w") as f:
        f.write(report_html)

    # Also write evo_progress.csv if provided
    if evo_history:
        _write_evo_progress_csv(output_dir, evo_history)

    return report_path
