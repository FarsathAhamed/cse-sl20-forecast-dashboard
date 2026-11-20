"""Reusable Plotly chart builders for the CSE S&P SL20 Forecast Dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_price_chart(
    df_price: pd.DataFrame,
    symbol: str,
    chart_type: str = "Candlestick",
    show_ma: bool = True,
) -> go.Figure:
    """
    Build an interactive OHLC candlestick or close-line chart in LKR
    with optional 10-day and 30-day moving averages and a volume subplot.
    """
    df = df_price.copy()
    if "MA10" not in df.columns:
        df["MA10"] = df["Close"].rolling(10).mean()
    if "MA30" not in df.columns:
        df["MA30"] = df["Close"].rolling(30).mean()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.74, 0.26],
    )

    if chart_type == "Candlestick":
        fig.add_trace(
            go.Candlestick(
                x=df["Date"],
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name=symbol,
                increasing_line_color="#16a34a",
                decreasing_line_color="#dc2626",
            ),
            row=1,
            col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["Close"],
                mode="lines",
                name=f"{symbol} Close",
                line=dict(color="#2563eb", width=2.2),
            ),
            row=1,
            col=1,
        )

    if show_ma:
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["MA10"],
                mode="lines",
                name="MA 10D",
                line=dict(color="#f59e0b", width=1.4, dash="solid"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["MA30"],
                mode="lines",
                name="MA 30D",
                line=dict(color="#8b5cf6", width=1.4, dash="dot"),
            ),
            row=1,
            col=1,
        )

    vol_colors = [
        "#16a34a" if c >= o else "#dc2626"
        for o, c in zip(df["Open"], df["Close"])
    ]
    fig.add_trace(
        go.Bar(
            x=df["Date"],
            y=df["Volume"],
            name="Volume",
            marker_color=vol_colors,
            opacity=0.65,
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=f"{symbol} — Historical Price (LKR) & Volume",
        height=540,
        margin=dict(l=16, r=16, t=48, b=16),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
    )
    fig.update_yaxes(title_text="Price (LKR)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


def build_confidence_gauge(prob_up: float, direction_label: str) -> go.Figure:
    """Build a clean horizontal probability bar comparing P(UP) vs P(NOT-UP)."""
    prob_not_up = max(0.0, min(1.0, 1.0 - prob_up))
    prob_up = max(0.0, min(1.0, prob_up))

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=["Next-Day Prob"],
            x=[prob_up * 100.0],
            name=f"P(UP): {prob_up:.1%}",
            orientation="h",
            marker=dict(color="#16a34a"),
            text=f"UP {prob_up:.1%}" if prob_up >= 0.14 else "",
            textposition="inside",
            insidetextanchor="middle",
            hovertemplate="P(UP): %{x:.1f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            y=["Next-Day Prob"],
            x=[prob_not_up * 100.0],
            name=f"P(NOT-UP): {prob_not_up:.1%}",
            orientation="h",
            marker=dict(color="#dc2626"),
            text=f"NOT-UP {prob_not_up:.1%}" if prob_not_up >= 0.14 else "",
            textposition="inside",
            insidetextanchor="middle",
            hovertemplate="P(NOT-UP): %{x:.1f}%<extra></extra>",
        )
    )

    fig.add_vline(x=50, line_width=2, line_dash="dash", line_color="#475569")

    fig.update_layout(
        title=f"Direction Probability Split — Predicted: {direction_label}",
        barmode="stack",
        height=135,
        margin=dict(l=16, r=16, t=42, b=16),
        xaxis=dict(range=[0, 100], ticksuffix="%", showgrid=False),
        yaxis=dict(showticklabels=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
    )
    return fig


def build_feature_importance_chart(perm_importance_dict: dict) -> go.Figure:
    """Build a sorted horizontal bar chart of hold-out permutation importance."""
    items = sorted(
        perm_importance_dict.items(),
        key=lambda kv: kv[1]["mean"],
    )
    features = [k for k, _ in items]
    means = [v["mean"] for _, v in items]
    stds = [v["std"] for _, v in items]
    colors = ["#2563eb" if m > 0 else "#94a3b8" for m in means]

    fig = go.Figure(
        go.Bar(
            y=features,
            x=means,
            orientation="h",
            marker_color=colors,
            error_x=dict(type="data", array=stds, visible=True, color="#475569", thickness=1.2),
            hovertemplate="%{y}: %{x:.4f} (±%{error_x.array:.4f})<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_width=1.5, line_dash="dash", line_color="#64748b")
    fig.update_layout(
        title="Hold-Out Permutation Importance (Balanced Accuracy Drop)",
        height=400,
        margin=dict(l=16, r=16, t=44, b=16),
        xaxis_title="Mean Importance (±1 SD across 30 repeats)",
        yaxis_title="Stationary Feature",
        template="plotly_white",
    )
    return fig


def build_confusion_matrix_chart(cm: list[list[int]], labels: list[str]) -> go.Figure:
    """Build a clean annotated Plotly confusion matrix heatmap."""
    total = sum(sum(row) for row in cm) or 1
    annotations = [
        [f"{val}<br>({val / total:.1%})" for val in row]
        for row in cm
    ]

    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            text=annotations,
            texttemplate="%{text}",
            textfont=dict(size=13),
            colorscale="Blues",
            showscale=False,
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Hold-Out Confusion Matrix (Direction RF)",
        height=400,
        margin=dict(l=16, r=16, t=44, b=16),
        xaxis_title="Predicted Class",
        yaxis_title="Actual Class",
        yaxis=dict(autorange="reversed"),
        template="plotly_white",
    )
    return fig


def build_all_stocks_comparison_chart(summary_df: pd.DataFrame) -> go.Figure:
    """
    Build a two-panel comparison across all 11 S&P SL20 stocks:
    (a) RF Direction Accuracy vs Majority Baseline
    (b) HAR Risk Out-of-Sample R^2 vs Historical Mean (highlighting Holm p < 0.05)
    """
    df_dir = summary_df.sort_values("rf_acc", ascending=True)
    df_risk = summary_df.sort_values("har_r2_vs_hist", ascending=True)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Direction: RF Accuracy vs. Majority Baseline (%)",
            "Risk (HAR): Out-of-Sample R² vs. Historical Mean",
        ),
        horizontal_spacing=0.12,
    )

    fig.add_trace(
        go.Bar(
            y=df_dir["symbol"],
            x=df_dir["rf_acc"] * 100.0,
            name="RF Accuracy (%)",
            orientation="h",
            marker_color="#2563eb",
            hovertemplate="%{y}: %{x:.1f}%<extra>RF Accuracy</extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            y=df_dir["symbol"],
            x=df_dir["maj_acc"] * 100.0,
            name="Majority Baseline (%)",
            orientation="h",
            marker_color="#cbd5e1",
            hovertemplate="%{y}: %{x:.1f}%<extra>Majority Baseline</extra>",
        ),
        row=1,
        col=1,
    )

    sig_mask = df_risk["har_dm_hist_p_holm"] < 0.05
    risk_colors = ["#16a34a" if sig else "#94a3b8" for sig in sig_mask]
    fig.add_trace(
        go.Bar(
            y=df_risk["symbol"],
            x=df_risk["har_r2_vs_hist"],
            name="HAR OOS R² (Green = Holm p < 0.05)",
            orientation="h",
            marker_color=risk_colors,
            hovertemplate="%{y}: R² = %{x:.3f}<extra>HAR Risk</extra>",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        height=460,
        margin=dict(l=16, r=16, t=56, b=16),
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5),
        template="plotly_white",
    )
    fig.update_xaxes(title_text="Hold-Out Accuracy (%)", range=[0, 75], row=1, col=1)
    fig.update_xaxes(title_text="Out-of-Sample R²", row=1, col=2)
    return fig
