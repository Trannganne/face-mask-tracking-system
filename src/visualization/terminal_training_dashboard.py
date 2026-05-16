import argparse
import json
import os
from pathlib import Path

import pandas as pd
import plotext as plt
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns
from rich.text import Text
from rich import box


console = Console()


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def normalize_col(name: str) -> str:
    return name.strip().replace(" ", "").lower()


def load_history_json(path: Path) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df.insert(0, "epoch", range(1, len(df) + 1))
    return df


def load_yolo_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "epoch" not in [normalize_col(c) for c in df.columns]:
        df.insert(0, "epoch", range(1, len(df) + 1))
    return df


def find_column(df: pd.DataFrame, candidates):
    normalized = {normalize_col(c): c for c in df.columns}
    for item in candidates:
        key = normalize_col(item)
        if key in normalized:
            return normalized[key]
    return None


def line_plot(title, x, series, width=76, height=18):
    plt.clear_figure()
    plt.theme("dark")
    plt.plotsize(width, height)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Value")

    for label, y in series:
        if y is not None:
            plt.plot(x, y, label=label)

    return plt.build()


def make_metric_table(title, metrics):
    table = Table(title=title, box=box.ROUNDED, show_header=True, header_style="bold")
    table.add_column("Metric", justify="left")
    table.add_column("Last", justify="right")
    table.add_column("Best", justify="right")

    for name, values, mode in metrics:
        if values is None or len(values) == 0:
            continue

        last_value = float(values.iloc[-1])
        best_value = float(values.max() if mode == "max" else values.min())

        table.add_row(
            name,
            f"{last_value:.4f}",
            f"{best_value:.4f}",
        )

    return table


def show_keras_dashboard(df: pd.DataFrame):
    epoch_col = "epoch"
    x = df[epoch_col]

    acc_col = find_column(df, ["accuracy", "acc", "train_accuracy", "train_acc"])
    val_acc_col = find_column(df, ["val_accuracy", "val_acc", "valid_accuracy", "valid_acc"])

    loss_col = find_column(df, ["loss", "train_loss"])
    val_loss_col = find_column(df, ["val_loss", "valid_loss"])

    title = Text("DEEP LEARNING TRAINING DASHBOARD", style="bold")
    subtitle = Text("Accuracy/Loss graph - xem trực tiếp trên Terminal", style="dim")

    console.print(Panel.fit(Text.assemble(title, "\n", subtitle), border_style="cyan"))

    overview = Table(title="Dataset / Training Overview", box=box.ROUNDED)
    overview.add_column("Item")
    overview.add_column("Value")
    overview.add_row("Total Epochs", str(len(df)))
    overview.add_row("Detected Type", "Keras/TensorFlow history.json")
    overview.add_row("Accuracy Column", acc_col or "Not found")
    overview.add_row("Validation Accuracy Column", val_acc_col or "Not found")
    overview.add_row("Loss Column", loss_col or "Not found")
    overview.add_row("Validation Loss Column", val_loss_col or "Not found")

    metrics = make_metric_table(
        "Final / Best Metrics",
        [
            ("Train Accuracy", df[acc_col] if acc_col else None, "max"),
            ("Val Accuracy", df[val_acc_col] if val_acc_col else None, "max"),
            ("Train Loss", df[loss_col] if loss_col else None, "min"),
            ("Val Loss", df[val_loss_col] if val_loss_col else None, "min"),
        ],
    )

    console.print(Columns([overview, metrics], equal=True, expand=True))

    if acc_col or val_acc_col:
        acc_plot = line_plot(
            "Accuracy Curve",
            x,
            [
                ("train_acc", df[acc_col] if acc_col else None),
                ("val_acc", df[val_acc_col] if val_acc_col else None),
            ],
        )
        console.print(Panel(acc_plot, title="Step 1 - Accuracy Graph", border_style="green"))

    if loss_col or val_loss_col:
        loss_plot = line_plot(
            "Loss Curve",
            x,
            [
                ("train_loss", df[loss_col] if loss_col else None),
                ("val_loss", df[val_loss_col] if val_loss_col else None),
            ],
        )
        console.print(Panel(loss_plot, title="Step 2 - Loss Graph", border_style="yellow"))

    console.print(Panel(
        "Cách đọc: Accuracy tăng và Loss giảm là dấu hiệu mô hình học tốt. "
        "Nếu train accuracy cao nhưng val accuracy thấp hoặc val loss tăng, mô hình có thể bị overfitting.",
        title="Nhận xét nhanh",
        border_style="magenta",
    ))


def show_yolo_dashboard(df: pd.DataFrame):
    epoch_col = find_column(df, ["epoch"]) or df.columns[0]
    x = df[epoch_col]

    precision_col = find_column(df, ["metrics/precision(B)", "precision", "metrics/precision"])
    recall_col = find_column(df, ["metrics/recall(B)", "recall", "metrics/recall"])
    map50_col = find_column(df, ["metrics/mAP50(B)", "map50", "mAP50"])
    map5095_col = find_column(df, ["metrics/mAP50-95(B)", "map50-95", "mAP50-95"])

    train_box_loss = find_column(df, ["train/box_loss", "box_loss"])
    train_cls_loss = find_column(df, ["train/cls_loss", "cls_loss"])
    train_dfl_loss = find_column(df, ["train/dfl_loss", "dfl_loss"])
    val_box_loss = find_column(df, ["val/box_loss"])
    val_cls_loss = find_column(df, ["val/cls_loss"])
    val_dfl_loss = find_column(df, ["val/dfl_loss"])

    title = Text("YOLO TRAINING DASHBOARD", style="bold")
    subtitle = Text("Precision / Recall / mAP / Loss - xem trực tiếp trên Terminal", style="dim")

    console.print(Panel.fit(Text.assemble(title, "\n", subtitle), border_style="cyan"))

    overview = Table(title="Training Overview", box=box.ROUNDED)
    overview.add_column("Item")
    overview.add_column("Value")
    overview.add_row("Total Epochs", str(len(df)))
    overview.add_row("Detected Type", "Ultralytics YOLO results.csv")
    overview.add_row("Precision", precision_col or "Not found")
    overview.add_row("Recall", recall_col or "Not found")
    overview.add_row("mAP@50", map50_col or "Not found")
    overview.add_row("mAP@50-95", map5095_col or "Not found")

    metrics = make_metric_table(
        "Final / Best Metrics",
        [
            ("Precision", df[precision_col] if precision_col else None, "max"),
            ("Recall", df[recall_col] if recall_col else None, "max"),
            ("mAP@50", df[map50_col] if map50_col else None, "max"),
            ("mAP@50-95", df[map5095_col] if map5095_col else None, "max"),
            ("Train Box Loss", df[train_box_loss] if train_box_loss else None, "min"),
            ("Train Cls Loss", df[train_cls_loss] if train_cls_loss else None, "min"),
            ("Train DFL Loss", df[train_dfl_loss] if train_dfl_loss else None, "min"),
        ],
    )

    console.print(Columns([overview, metrics], equal=True, expand=True))

    metric_series = [
        ("precision", df[precision_col] if precision_col else None),
        ("recall", df[recall_col] if recall_col else None),
        ("mAP50", df[map50_col] if map50_col else None),
        ("mAP50-95", df[map5095_col] if map5095_col else None),
    ]

    if any(y is not None for _, y in metric_series):
        metric_plot = line_plot("YOLO Metrics Curve", x, metric_series)
        console.print(Panel(metric_plot, title="Step 1 - Precision / Recall / mAP", border_style="green"))

    loss_series = [
        ("train_box", df[train_box_loss] if train_box_loss else None),
        ("train_cls", df[train_cls_loss] if train_cls_loss else None),
        ("train_dfl", df[train_dfl_loss] if train_dfl_loss else None),
        ("val_box", df[val_box_loss] if val_box_loss else None),
        ("val_cls", df[val_cls_loss] if val_cls_loss else None),
        ("val_dfl", df[val_dfl_loss] if val_dfl_loss else None),
    ]

    if any(y is not None for _, y in loss_series):
        loss_plot = line_plot("YOLO Loss Curve", x, loss_series)
        console.print(Panel(loss_plot, title="Step 2 - YOLO Loss Graph", border_style="yellow"))

    console.print(Panel(
        "Lưu ý: YOLO thường không dùng Accuracy truyền thống. "
        "Nên trình bày Precision, Recall, mAP@50, mAP@50-95 và Loss trong báo cáo.",
        title="Nhận xét nhanh",
        border_style="magenta",
    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to history.json or YOLO results.csv")
    parser.add_argument("--no-clear", action="store_true", help="Do not clear terminal before showing dashboard")
    args = parser.parse_args()

    path = Path(args.input)

    if not path.exists():
        console.print(f"[bold red]Không tìm thấy file:[/] {path}")
        console.print("Ví dụ:")
        console.print("  python terminal_training_dashboard.py --input models/train_logs/history.json")
        console.print("  python terminal_training_dashboard.py --input runs/detect/train/results.csv")
        return

    if not args.no_clear:
        clear_screen()

    if path.suffix.lower() == ".json":
        df = load_history_json(path)
        show_keras_dashboard(df)
    elif path.suffix.lower() == ".csv":
        df = load_yolo_csv(path)
        show_yolo_dashboard(df)
    else:
        console.print("[bold red]File không hỗ trợ. Chỉ hỗ trợ .json hoặc .csv[/]")


if __name__ == "__main__":
    main()
