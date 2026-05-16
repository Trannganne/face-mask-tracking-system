import argparse
import pandas as pd
import plotext as plt
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns
from rich import box

console = Console()


def load_data(path):
    return pd.read_csv(path)


def draw_plot(title, x, y1, y2, label1, label2):
    plt.clear_figure()
    plt.theme("dark")
    plt.plotsize(80, 20)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.plot(x, y1, label=label1)
    plt.plot(x, y2, label=label2)
    return plt.build()


def show_dashboard(df):
    console.print(
        Panel.fit(
            "[bold cyan]FACE MASK DETECTION TRAINING DASHBOARD[/bold cyan]\n"
            "[dim]Accuracy / Loss graph - Terminal UI[/dim]",
            border_style="cyan"
        )
    )

    total_epochs = len(df)

    train_acc = df["accuracy"]
    val_acc = df["val_accuracy"]
    train_loss = df["train_loss"]
    val_loss = df["val_loss"]

    overview = Table(title="Training Overview", box=box.ROUNDED)
    overview.add_column("Item", style="cyan")
    overview.add_column("Value", style="green")

    overview.add_row("Project", "Face Mask Detection")
    overview.add_row("Mode", "Terminal Dashboard")
    overview.add_row("Total Epochs", str(total_epochs))
    overview.add_row("Data Source", "CSV Result File")

    metrics = Table(title="Final / Best Metrics", box=box.ROUNDED)
    metrics.add_column("Metric", style="cyan")
    metrics.add_column("Last", justify="right", style="yellow")
    metrics.add_column("Best", justify="right", style="green")

    metrics.add_row("Train Accuracy", f"{train_acc.iloc[-1]:.4f}", f"{train_acc.max():.4f}")
    metrics.add_row("Val Accuracy", f"{val_acc.iloc[-1]:.4f}", f"{val_acc.max():.4f}")
    metrics.add_row("Train Loss", f"{train_loss.iloc[-1]:.4f}", f"{train_loss.min():.4f}")
    metrics.add_row("Val Loss", f"{val_loss.iloc[-1]:.4f}", f"{val_loss.min():.4f}")

    console.print(Columns([overview, metrics], equal=True, expand=True))

    acc_graph = draw_plot(
        "Accuracy Curve",
        df["epoch"],
        train_acc,
        val_acc,
        "Train Accuracy",
        "Validation Accuracy"
    )

    console.print(
        Panel(
            acc_graph,
            title="[bold green]Step 1 - Accuracy Graph[/bold green]",
            border_style="green"
        )
    )

    loss_graph = draw_plot(
        "Loss Curve",
        df["epoch"],
        train_loss,
        val_loss,
        "Train Loss",
        "Validation Loss"
    )

    console.print(
        Panel(
            loss_graph,
            title="[bold yellow]Step 2 - Loss Graph[/bold yellow]",
            border_style="yellow"
        )
    )

    console.print(
        Panel(
            "Accuracy tăng dần và Loss giảm dần cho thấy mô hình đang học tốt. "
            "Nếu Train Accuracy cao nhưng Validation Accuracy thấp, mô hình có thể bị overfitting.",
            title="[bold magenta]Nhận xét nhanh[/bold magenta]",
            border_style="magenta"
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to CSV file")
    args = parser.parse_args()

    df = load_data(args.input)
    show_dashboard(df)


if __name__ == "__main__":
    main()