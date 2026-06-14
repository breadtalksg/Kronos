#!/usr/bin/env python3
"""Forecast a public Binance K-line series with a pretrained Kronos model."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import requests
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import Kronos, KronosPredictor, KronosTokenizer


BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SUPPORTED_INTERVALS = {
    "1m": pd.Timedelta(minutes=1),
    "3m": pd.Timedelta(minutes=3),
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "30m": pd.Timedelta(minutes=30),
    "1h": pd.Timedelta(hours=1),
    "2h": pd.Timedelta(hours=2),
    "4h": pd.Timedelta(hours=4),
    "6h": pd.Timedelta(hours=6),
    "8h": pd.Timedelta(hours=8),
    "12h": pd.Timedelta(hours=12),
    "1d": pd.Timedelta(days=1),
    "3d": pd.Timedelta(days=3),
    "1w": pd.Timedelta(weeks=1),
    "1M": pd.DateOffset(months=1),
}
MODEL_CONFIGS = {
    "mini": {
        "model_id": "NeoQuasar/Kronos-mini",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-2k",
        "max_context": 2048,
    },
    "small": {
        "model_id": "NeoQuasar/Kronos-small",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
        "max_context": 512,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Forecast public Binance K-line data with Kronos."
    )
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1h", choices=SUPPORTED_INTERVALS)
    parser.add_argument("--lookback", type=int, default=500)
    parser.add_argument("--pred-steps", type=int, default=24)
    parser.add_argument("--model", default="mini", choices=MODEL_CONFIGS)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.symbol.isalnum():
        raise ValueError("--symbol must contain only letters and numbers.")
    if args.lookback < 2:
        raise ValueError("--lookback must be at least 2.")
    if args.pred_steps < 1:
        raise ValueError("--pred-steps must be at least 1.")


def fetch_binance_klines(symbol: str, interval: str, lookback: int) -> pd.DataFrame:
    """Fetch the latest completed public K-lines, paginating when necessary."""
    rows: list[list[object]] = []
    end_time: int | None = None
    target_rows = lookback + 1

    with requests.Session() as session:
        while len(rows) < target_rows:
            request_limit = min(1000, target_rows - len(rows))
            params: dict[str, object] = {
                "symbol": symbol.upper(),
                "interval": interval,
                "limit": request_limit,
            }
            if end_time is not None:
                params["endTime"] = end_time

            response = session.get(BINANCE_KLINES_URL, params=params, timeout=30)
            response.raise_for_status()
            batch = response.json()
            if not isinstance(batch, list) or not batch:
                break

            rows = batch + rows
            end_time = int(batch[0][0]) - 1
            if len(batch) < request_limit:
                break

    now_ms = int(time.time() * 1000)
    completed_rows = [row for row in rows if int(row[6]) < now_ms]
    completed_rows = list({int(row[0]): row for row in completed_rows}.values())
    completed_rows.sort(key=lambda row: int(row[0]))
    completed_rows = completed_rows[-lookback:]

    if len(completed_rows) < lookback:
        raise RuntimeError(
            f"Binance returned only {len(completed_rows)} completed K-lines; "
            f"{lookback} were requested."
        )

    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [row[0] for row in completed_rows], unit="ms", utc=True
            ),
            "open": [row[1] for row in completed_rows],
            "high": [row[2] for row in completed_rows],
            "low": [row[3] for row in completed_rows],
            "close": [row[4] for row in completed_rows],
            "volume": [row[5] for row in completed_rows],
            "amount": [row[7] for row in completed_rows],
        }
    )
    value_columns = ["open", "high", "low", "close", "volume", "amount"]
    frame[value_columns] = frame[value_columns].apply(pd.to_numeric, errors="raise")
    return frame


def resolve_device(requested_device: str) -> str:
    if requested_device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    return requested_device


def build_future_timestamps(
    last_timestamp: pd.Timestamp, interval: str, pred_steps: int
) -> pd.Series:
    offset = SUPPORTED_INTERVALS[interval]
    return pd.Series(
        pd.date_range(
            start=last_timestamp + offset,
            periods=pred_steps,
            freq=offset,
        )
    )


def load_predictor(model_name: str, device: str) -> KronosPredictor:
    config = MODEL_CONFIGS[model_name]
    print(f"Loading {config['model_id']} on {device}...")
    tokenizer = KronosTokenizer.from_pretrained(config["tokenizer_id"]).eval()
    model = Kronos.from_pretrained(config["model_id"]).eval()
    return KronosPredictor(
        model,
        tokenizer,
        device=device,
        max_context=config["max_context"],
    )


def format_forecast(prediction: pd.DataFrame) -> pd.DataFrame:
    output = prediction.reset_index(names="timestamp").rename(
        columns={
            "open": "pred_open",
            "high": "pred_high",
            "low": "pred_low",
            "close": "pred_close",
            "volume": "pred_volume",
        }
    )
    output["scenario"] = "base"
    output["confidence_note"] = "single_path_forecast_only"
    return output[
        [
            "timestamp",
            "pred_open",
            "pred_high",
            "pred_low",
            "pred_close",
            "pred_volume",
            "scenario",
            "confidence_note",
        ]
    ]


def save_plot(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    symbol: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        history["timestamp"],
        history["close"],
        label="Historical",
        color="#2563eb",
        linewidth=1.5,
    )
    ax.plot(
        forecast["timestamp"],
        forecast["pred_close"],
        label="Forecast",
        color="#dc2626",
        linewidth=1.8,
    )
    ax.set_title(f"{symbol.upper()} Kronos Forecast")
    ax.set_xlabel("Timestamp (UTC)")
    ax.set_ylabel("Close Price")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.text(
        0.5,
        0.01,
        "Research only. Not financial advice.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = resolve_device(args.device)

    print(
        f"Fetching {args.lookback} completed {args.interval} K-lines "
        f"for {args.symbol.upper()}..."
    )
    history = fetch_binance_klines(args.symbol, args.interval, args.lookback)
    future_timestamps = build_future_timestamps(
        history["timestamp"].iloc[-1], args.interval, args.pred_steps
    )
    predictor = load_predictor(args.model, device)

    prediction = predictor.predict(
        df=history[["open", "high", "low", "close", "volume", "amount"]],
        x_timestamp=history["timestamp"],
        y_timestamp=future_timestamps,
        pred_len=args.pred_steps,
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=True,
    )
    forecast = format_forecast(prediction)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "btcusdt_forecast.csv"
    plot_path = OUTPUT_DIR / "btcusdt_forecast.png"
    forecast.to_csv(csv_path, index=False)
    save_plot(history, forecast, args.symbol, plot_path)

    print(f"Forecast CSV: {csv_path}")
    print(f"Forecast chart: {plot_path}")
    print("Research only. Not financial advice.")


if __name__ == "__main__":
    main()
