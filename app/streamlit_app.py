"""Main Streamlit application for the CSE S&P SL20 Forecast & Risk Dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.charts import (  # noqa: E402
    build_all_stocks_comparison_chart,
    build_confidence_gauge,
    build_confusion_matrix_chart,
    build_feature_importance_chart,
    build_price_chart,
)
from app.predictor import (  # noqa: E402
    get_latest_forecast,
    get_stock_catalog,
    load_all_stocks_summary,
    load_price_history,
    train_or_load_models,
)

st.set_page_config(
    page_title="CSE S&P SL20 Forecast & Risk Dashboard",
    page_icon="📈",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def cached_catalog() -> list[dict[str, str]]:
    return get_stock_catalog()


@st.cache_resource(show_spinner=False)
def cached_models(symbol: str) -> dict:
    return train_or_load_models(symbol)


@st.cache_data(show_spinner=False)
def cached_price_history(symbol: str) -> pd.DataFrame:
    df = load_price_history(symbol)
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA30"] = df["Close"].rolling(30).mean()
    return df


@st.cache_data(show_spinner=False)
def cached_forecast(symbol: str) -> dict:
    _ = cached_models(symbol)
    return get_latest_forecast(symbol)


@st.cache_data(show_spinner=False)
def cached_all_summary() -> pd.DataFrame:
    return load_all_stocks_summary()


def _format_p(value: float) -> str:
    if pd.isna(value):
        return "—"
    if value < 0.0001:
        return "< 0.0001"
    return f"{value:.4f}"


def _filter_by_window(df: pd.DataFrame, window: str) -> pd.DataFrame:
    if window == "All" or df.empty:
        return df
    end_date = df["Date"].max()
    offsets = {
        "3M": pd.DateOffset(months=3),
        "6M": pd.DateOffset(months=6),
        "1Y": pd.DateOffset(years=1),
        "3Y": pd.DateOffset(years=3),
    }
    start_date = end_date - offsets[window]
    return df.loc[df["Date"] >= start_date].copy()


def _render_forecast_tab(forecast: dict, df_window: pd.DataFrame, symbol: str, chart_type: str, show_ma: bool) -> None:
    dir_m = forecast["frozen_metrics"]["direction"]
    risk_m = forecast["frozen_metrics"]["risk"]
    summary_row = forecast["summary_row"]
    dir_p_holm = summary_row.get("dir_p_holm", float("nan"))
    har_p_holm = summary_row.get("har_dm_hist_p_holm", float("nan"))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        label=f"Latest Close ({forecast['latest_date']})",
        value=f"LKR {forecast['latest_close_lkr']:,.2f}",
        delta=f"{forecast['daily_change_pct']:+.2f}%",
        help="Most recent cleaned closing price in Sri Lankan Rupees (LKR) and 1-day percentage change.",
    )
    col2.metric(
        label="Next-Day Direction",
        value=forecast["direction_label"],
        delta=f"P(UP): {forecast['prob_up']:.1%}",
        delta_color="off",
        help="Next-day directional signal from the leakage-controlled Random Forest classifier.",
    )
    col3.metric(
        label="Model Confidence",
        value=f"{forecast['confidence_score']:.1%}",
        delta=f"Hold-out Acc: {dir_m['rf_test_acc']:.1%}",
        delta_color="off",
        help="Predicted class probability max(P(UP), P(NOT-UP)) alongside frozen hold-out accuracy.",
    )
    col4.metric(
        label="Expected Next-Day Risk (HAR)",
        value=f"{forecast['predicted_risk_range_pct']:.2f}%",
        delta=f"Risk Tier: {forecast['risk_level']}",
        delta_color="off",
        help="HAR-OLS forecast of next-day intraday range (High - Low) / Close, tiered by training quantiles.",
    )

    st.plotly_chart(
        build_confidence_gauge(forecast["prob_up"], forecast["direction_label"]),
        width="stretch",
    )

    st.plotly_chart(
        build_price_chart(df_window, symbol=symbol, chart_type=chart_type, show_ma=show_ma),
        width="stretch",
    )

    st.info(
        f"**Research Transparency Note (Not Financial Advice):** "
        f"This dashboard runs a leakage-controlled academic prototype for **{forecast['name']} ({symbol})** "
        f"trained chronologically on {dir_m['n_train']} days ({dir_m['train_start']} to {dir_m['train_end']}) "
        f"with a 10-day embargo gap and evaluated on {dir_m['n_test']} hold-out days. "
        f"On hold-out, Random Forest direction accuracy is **{dir_m['rf_test_acc']:.1%}** vs. Majority baseline "
        f"**{dir_m['maj_acc_train_rule']:.1%}** (Holm *p* = {_format_p(dir_p_holm)}), reflecting near-random daily direction. "
        f"By contrast, HAR volatility forecasting achieves out-of-sample *R*² = **{risk_m['har_r2_vs_hist']:.3f}** "
        f"vs. historical mean (Diebold–Mariano Holm *p* = {_format_p(har_p_holm)})."
    )


def _render_insights_tab(forecast: dict, symbol: str) -> None:
    dir_m = forecast["frozen_metrics"]["direction"]
    risk_m = forecast["frozen_metrics"]["risk"]
    summary_row = forecast["summary_row"]

    st.subheader(f"{forecast['name']} ({symbol}) — Hold-Out Benchmarks")
    left, right = st.columns(2)

    with left:
        st.markdown("**Direction Forecasting (H1 — Next-Day UP vs. NOT-UP)**")
        dir_table = pd.DataFrame(
            [
                {"Model": "Random Forest (Tuned)", "Hold-Out Accuracy": f"{dir_m['rf_test_acc']:.2%}", "Balanced Acc": f"{dir_m['rf_bal_acc']:.2%}", "MCC": f"{dir_m['rf_mcc']:.3f}"},
                {"Model": "Majority Class Baseline", "Hold-Out Accuracy": f"{dir_m['maj_acc_train_rule']:.2%}", "Balanced Acc": "50.00%", "MCC": "0.000"},
                {"Model": "Previous-Day Direction", "Hold-Out Accuracy": f"{dir_m['prev_day_acc']:.2%}", "Balanced Acc": f"{dir_m['prev_day_bal_acc']:.2%}", "MCC": f"{dir_m['prev_day_mcc']:.3f}"},
                {"Model": "Logistic Regression", "Hold-Out Accuracy": f"{dir_m['lr_acc']:.2%}", "Balanced Acc": f"{dir_m['lr_bal_acc']:.2%}", "MCC": f"{dir_m['lr_mcc']:.3f}"},
            ]
        )
        st.dataframe(dir_table, width="stretch", hide_index=True)
        st.caption(
            f"Wilson 95% CI: [{dir_m['rf_acc_ci_low']:.1%}, {dir_m['rf_acc_ci_high']:.1%}] | "
            f"Binomial raw p = {_format_p(dir_m['dir_p_raw'])} | "
            f"Holm-adjusted p = {_format_p(summary_row.get('dir_p_holm', float('nan')))}"
        )

    with right:
        st.markdown("**Volatility / Risk Forecasting (H2 — Next-Day Intraday Range)**")
        risk_table = pd.DataFrame(
            [
                {"Model": "HAR (OLS)", "OOS R² vs. Hist Mean": f"{risk_m['har_r2_vs_hist']:.4f}", "OOS R² vs. 22D Mean": f"{risk_m['har_r2_vs_m22']:.4f}", "Spearman ρ": f"{risk_m['spearman_har']:.3f}"},
                {"Model": "Random Forest Regressor", "OOS R² vs. Hist Mean": f"{risk_m['rf_r2_vs_hist']:.4f}", "OOS R² vs. 22D Mean": f"{risk_m['rf_r2_vs_m22']:.4f}", "Spearman ρ": f"{risk_m['spearman_rf']:.3f}"},
                {"Model": "22-Day Rolling Mean", "OOS R² vs. Hist Mean": f"{risk_m['m22_r2_vs_hist']:.4f}", "OOS R² vs. 22D Mean": "0.0000", "Spearman ρ": f"{risk_m['spearman_m22']:.3f}"},
                {"Model": "Previous-Day Range", "OOS R² vs. Hist Mean": f"{risk_m['prev_r2_vs_hist']:.4f}", "OOS R² vs. 22D Mean": "—", "Spearman ρ": "—"},
            ]
        )
        st.dataframe(risk_table, width="stretch", hide_index=True)
        st.caption(
            f"HAR vs. Hist DM stat = {risk_m['dm_har_vs_hist']['dm_stat']:.2f} "
            f"(Holm p = {_format_p(summary_row.get('har_dm_hist_p_holm', float('nan')))}) | "
            f"HAR vs. 22D DM Holm p = {_format_p(summary_row.get('har_dm_m22_p_holm', float('nan')))}"
        )

    col_fi, col_cm = st.columns([1.3, 1.0])
    with col_fi:
        st.plotly_chart(
            build_feature_importance_chart(dir_m["permutation_importance"]),
            width="stretch",
        )
    with col_cm:
        st.plotly_chart(
            build_confusion_matrix_chart(dir_m["confusion_matrix"], dir_m["confusion_matrix_labels"]),
            width="stretch",
        )

    st.markdown(
        "**Plain-English Statistical Interpretation:**\n"
        "- **Direction (Binomial + Holm):** Tests whether the Random Forest beats the Majority Class rule significantly across all 11 stocks. A Holm-adjusted *p* ≥ 0.05 means daily price direction behaves close to a random walk once look-ahead leakage is removed.\n"
        "- **Risk (Diebold–Mariano with Newey–West HAC + Holm):** Tests whether the 3-feature HAR model has significantly lower squared forecast error than historical or 22-day averages while accounting for autocorrelation in financial volatility."
    )


def _render_overview_tab(summary_df: pd.DataFrame) -> None:
    st.subheader("S&P SL20 Cross-Stock Benchmark Summary (11 Stocks)")

    n_dir_sig = int((summary_df["dir_p_holm"] < 0.05).sum())
    n_risk_sig = int((summary_df["har_dm_hist_p_holm"] < 0.05).sum())
    median_har_r2 = float(summary_df["har_r2_vs_hist"].median())

    m1, m2, m3 = st.columns(3)
    m1.metric("Direction Significant (Holm p < 0.05)", f"{n_dir_sig} / {len(summary_df)} stocks")
    m2.metric("HAR Risk Significant vs. Hist Mean", f"{n_risk_sig} / {len(summary_df)} stocks")
    m3.metric("Median HAR Out-of-Sample R²", f"{median_har_r2:.3f}")

    st.plotly_chart(
        build_all_stocks_comparison_chart(summary_df),
        width="stretch",
    )

    display_df = summary_df[
        [
            "symbol",
            "name",
            "n_train_dir",
            "n_test_dir",
            "rf_acc",
            "maj_acc",
            "rf_bal_acc",
            "dir_p_holm",
            "har_r2_vs_hist",
            "har_dm_hist_p_holm",
            "spearman_har",
        ]
    ].rename(
        columns={
            "symbol": "Ticker",
            "name": "Company",
            "n_train_dir": "Train Days",
            "n_test_dir": "Test Days",
            "rf_acc": "RF Acc",
            "maj_acc": "Majority Acc",
            "rf_bal_acc": "RF Bal Acc",
            "dir_p_holm": "Dir Holm p",
            "har_r2_vs_hist": "HAR OOS R²",
            "har_dm_hist_p_holm": "HAR DM Holm p",
            "spearman_har": "HAR Spearman ρ",
        }
    )

    st.dataframe(
        display_df.style.format(
            {
                "RF Acc": "{:.2%}",
                "Majority Acc": "{:.2%}",
                "RF Bal Acc": "{:.2%}",
                "Dir Holm p": "{:.4f}",
                "HAR OOS R²": "{:.4f}",
                "HAR DM Holm p": "{:.4f}",
                "HAR Spearman ρ": "{:.3f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    st.download_button(
        label="Download Full 11-Stock Summary (CSV)",
        data=summary_df.to_csv(index=False).encode("utf-8"),
        file_name="cse_sp_sl20_summary_11_stocks.csv",
        mime="text/csv",
    )


def main() -> None:
    st.title("CSE S&P SL20 Forecast & Risk Dashboard")
    st.caption(
        "Leakage-controlled next-day direction & HAR volatility evaluation across 11 Colombo Stock Exchange (S&P SL20) equities."
    )

    catalog = cached_catalog()
    labels = {f"{item['symbol']} — {item['name']}": item["symbol"] for item in catalog}

    with st.sidebar:
        st.header("Controls")
        selected_label = st.selectbox("Select S&P SL20 Stock", list(labels.keys()), index=6)
        symbol = labels[selected_label]

        window = st.radio(
            "Price Chart Window",
            options=["3M", "6M", "1Y", "3Y", "All"],
            index=2,
            horizontal=True,
        )
        chart_type = st.radio("Chart Style", options=["Candlestick", "Line"], index=0, horizontal=True)
        show_ma = st.toggle("Show 10D & 30D Moving Averages", value=True)

    forecast = cached_forecast(symbol)
    df_price = cached_price_history(symbol)
    df_window = _filter_by_window(df_price, window)
    summary_df = cached_all_summary()

    tab1, tab2, tab3 = st.tabs(
        [
            "Forecast & Price Chart",
            "Model Insights & Validation",
            "S&P SL20 Overview (11 Stocks)",
        ]
    )

    with tab1:
        _render_forecast_tab(forecast, df_window, symbol, chart_type, show_ma)
    with tab2:
        _render_insights_tab(forecast, symbol)
    with tab3:
        _render_overview_tab(summary_df)


if __name__ == "__main__":
    main()
