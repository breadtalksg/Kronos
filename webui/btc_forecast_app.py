#!/usr/bin/env python3
"""Streamlit dashboard for the research-only BTCUSDT Kronos forecast."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.forecast_btcusdt import (
    OUTPUT_DIR,
    SINGAPORE_TIMEZONE,
    build_forecast_figure,
    build_future_timestamps,
    fetch_binance_klines,
    format_forecast,
    load_predictor,
    resolve_device,
    save_plot,
    to_timezone,
)


st.set_page_config(
    page_title="BTCUSDT Kronos Forecast Dashboard",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_predictor(model_name: str, device: str):
    return load_predictor(model_name, device)


def calculate_summary(
    history: pd.DataFrame, forecast: pd.DataFrame, interval: str
) -> dict[str, object]:
    current_price = float(history["close"].iloc[-1])
    final_close = float(forecast["pred_close"].iloc[-1])
    forecast_high = float(forecast["pred_high"].max())
    forecast_low = float(forecast["pred_low"].min())
    change_pct = (final_close / current_price - 1) * 100
    forecast_range = max(forecast_high - forecast_low, 0.0)
    support = (forecast_low, forecast_low + forecast_range * 0.2)
    resistance = (forecast_high - forecast_range * 0.2, forecast_high)

    if change_pct > 1:
        direction = "偏多"
    elif change_pct < -1:
        direction = "偏弱"
    else:
        direction = "偏震荡"

    interval_names = {"1h": "1小时", "4h": "4小时", "1d": "日"}
    return {
        "current_price": current_price,
        "final_close": final_close,
        "forecast_high": forecast_high,
        "forecast_low": forecast_low,
        "change_pct": change_pct,
        "support": support,
        "resistance": resistance,
        "direction": direction,
        "interval_name": interval_names[interval],
    }


def format_price(value: float) -> str:
    return f"{value:,.2f}"


def run_forecast(
    symbol: str,
    interval: str,
    lookback: int,
    pred_steps: int,
    model_name: str,
    requested_device: str,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    device = resolve_device(requested_device)
    history = fetch_binance_klines(symbol, interval, lookback)
    future_timestamps = build_future_timestamps(
        history["timestamp"].iloc[-1], interval, pred_steps
    )
    predictor = get_predictor(model_name, device)
    prediction = predictor.predict(
        df=history[["open", "high", "low", "close", "volume", "amount"]],
        x_timestamp=history["timestamp"],
        y_timestamp=future_timestamps,
        pred_len=pred_steps,
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=False,
    )
    return history, format_forecast(prediction), device


st.title("BTCUSDT Kronos Forecast Dashboard")
st.caption("公开 Binance K 线数据 · Singapore Time / UTC+8 · 仅供研究参考")

with st.sidebar:
    st.header("Forecast Parameters")
    symbol = st.text_input("Symbol", value="BTCUSDT").strip().upper()
    interval = st.selectbox("Interval", ["1h", "4h", "1d"], index=0)
    lookback = st.number_input("Lookback", min_value=50, value=500, step=50)
    pred_steps = st.number_input("Predict Steps", min_value=1, value=24, step=1)
    model_name = st.selectbox("Model", ["mini", "small"], index=0)
    requested_device = st.selectbox("Device", ["auto", "cpu", "cuda"], index=0)
    run_clicked = st.button("Run Forecast", type="primary", width="stretch")

if not run_clicked and "btc_forecast_result" not in st.session_state:
    st.info("设置参数后，点击 Run Forecast 开始预测。")
    st.stop()

if run_clicked:
    try:
        with st.spinner("正在读取公开行情、加载 Kronos 模型并生成预测..."):
            history, forecast, resolved_device = run_forecast(
                symbol,
                interval,
                int(lookback),
                int(pred_steps),
                model_name,
                requested_device,
            )

            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            csv_path = OUTPUT_DIR / "btcusdt_forecast.csv"
            plot_path = OUTPUT_DIR / "btcusdt_forecast.png"
            forecast.to_csv(csv_path, index=False)
            save_plot(
                history,
                forecast,
                symbol,
                plot_path,
                timezone=SINGAPORE_TIMEZONE,
                title=f"{symbol} Kronos Forecast - Singapore Time",
            )
            st.session_state["btc_forecast_result"] = {
                "history": history,
                "forecast": forecast,
                "device": resolved_device,
                "params": {
                    "symbol": symbol,
                    "interval": interval,
                    "lookback": int(lookback),
                    "pred_steps": int(pred_steps),
                    "model_name": model_name,
                },
            }
    except Exception as exc:
        st.error(f"预测运行失败：{exc}")
        st.stop()

result = st.session_state["btc_forecast_result"]
history = result["history"]
forecast = result["forecast"]
resolved_device = result["device"]
symbol = result["params"]["symbol"]
interval = result["params"]["interval"]
lookback = result["params"]["lookback"]
pred_steps = result["params"]["pred_steps"]
model_name = result["params"]["model_name"]
summary = calculate_summary(history, forecast, interval)

st.subheader("当前参数")
st.json(
    {
        "symbol": symbol,
        "interval": interval,
        "lookback": int(lookback),
        "predict_steps": int(pred_steps),
        "model": model_name,
        "device": resolved_device,
        "timezone": "Singapore Time / UTC+8",
    }
)

first_row = st.columns(5)
first_row[0].metric("最新历史价格", format_price(summary["current_price"]))
first_row[1].metric(
    "预测最后收盘价",
    format_price(summary["final_close"]),
    f"{summary['change_pct']:+.2f}%",
)
first_row[2].metric("预测最高价", format_price(summary["forecast_high"]))
first_row[3].metric("预测最低价", format_price(summary["forecast_low"]))
first_row[4].metric("预测涨跌幅", f"{summary['change_pct']:+.2f}%")

second_row = st.columns(2)
second_row[0].metric(
    "支撑区",
    f"{format_price(summary['support'][0])} - {format_price(summary['support'][1])}",
)
second_row[1].metric(
    "压力区",
    f"{format_price(summary['resistance'][0])} - "
    f"{format_price(summary['resistance'][1])}",
)

last_history_time = to_timezone(
    pd.Series([history["timestamp"].iloc[-1]]), SINGAPORE_TIMEZONE
).iloc[0]
last_forecast_time = forecast["timestamp_sgt"].iloc[-1]
st.subheader("中文总结")
st.markdown(
    f"""
Kronos 本次预测 **{symbol}** 未来 **{int(pred_steps)} 根{summary['interval_name']}K线**
整体 **{summary['direction']}**。

- 最新历史价格：**{format_price(summary['current_price'])}**
  （{last_history_time.strftime('%Y-%m-%d %H:%M:%S %Z')}）
- 预测区间大约为：**{format_price(summary['forecast_low'])} - {format_price(summary['forecast_high'])}**
- 支撑区：**{format_price(summary['support'][0])} - {format_price(summary['support'][1])}**
- 压力区：**{format_price(summary['resistance'][0])} - {format_price(summary['resistance'][1])}**
- 最后预测收盘价：**{format_price(summary['final_close'])}**
  （{last_forecast_time.strftime('%Y-%m-%d %H:%M:%S %Z')}）
- 相对当前价格预测涨跌幅：**{summary['change_pct']:+.2f}%**

**仅供研究参考，不构成投资建议。**
"""
)

st.subheader("预测图表")
figure = build_forecast_figure(
    history,
    forecast,
    symbol,
    timezone=SINGAPORE_TIMEZONE,
    title=f"{symbol} Kronos Forecast - Singapore Time",
)
st.pyplot(figure, width="stretch")

st.subheader("预测 CSV 表格")
display_columns = [
    "timestamp_sgt",
    "pred_open",
    "pred_high",
    "pred_low",
    "pred_close",
    "pred_volume",
    "scenario",
    "confidence_note",
]
st.dataframe(forecast[display_columns], width="stretch", hide_index=True)
st.download_button(
    "Download Forecast CSV",
    data=forecast.to_csv(index=False).encode("utf-8"),
    file_name="btcusdt_forecast.csv",
    mime="text/csv",
)

st.caption("Research only. Not financial advice.")
