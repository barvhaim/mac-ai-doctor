from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .estimate import DTYPE_BYTES, estimate
from .metadata import resolve_model
from .models import Estimate, Verdict
from .system import detect_system

app = typer.Typer(no_args_is_help=True, help="Will this AI model fit in your Mac's unified memory?")
console = Console()


def _memory(value: float | None) -> float:
    detected = detect_system()
    result = value if value is not None else detected.memory_gb
    if result is None:
        raise typer.BadParameter("cannot detect unified memory here; pass --memory-gb N")
    if result <= 0:
        raise typer.BadParameter("memory must be greater than zero")
    return result


VERDICT_COLOR = {
    Verdict.COMFORTABLE: "green",
    Verdict.TIGHT: "yellow",
    Verdict.UNLIKELY: "red",
    Verdict.UNKNOWN: "dim",
}
VERDICT_GUIDANCE = {
    Verdict.COMFORTABLE: "Good fit. Keep normal apps modest for best stability.",
    Verdict.TIGHT: "May fit, but close memory-heavy apps or reduce context/concurrency.",
    Verdict.UNLIKELY: "Choose a smaller/more-quantized model or reduce context.",
    Verdict.UNKNOWN: "Provide a model with weight-size metadata for a fit verdict.",
}
DISCLAIMER = "Estimate, not a benchmark. No token/s prediction is made."


def build_check_renderables(item: Estimate) -> tuple[Table, Panel, str]:
    """Build the shared table, verdict panel, and disclaimer for a single estimate."""
    color = VERDICT_COLOR[item.verdict]
    table = Table(title=item.model.model_id, show_header=False)
    table.add_column("Component", style="cyan")
    table.add_column("Estimate", justify="right")
    table.add_row(
        "Weights", f"{item.weights_gb:.2f} GB" if item.weights_gb is not None else "unknown"
    )
    table.add_row(
        "KV cache", f"{item.kv_cache_gb:.2f} GB" if item.kv_cache_gb is not None else "unknown"
    )
    table.add_row(
        "Runtime", f"{item.runtime_gb:.2f} GB" if item.runtime_gb is not None else "unknown"
    )
    table.add_row(
        "Peak range",
        f"{item.low_gb:.2f}–{item.high_gb:.2f} GB" if item.low_gb is not None else "unknown",
    )
    table.add_row("Available", f"{item.available_gb:.1f} GB")
    panel = Panel(
        VERDICT_GUIDANCE[item.verdict],
        title=f"[{color}]{item.verdict.value}[/{color}] · {item.confidence} confidence",
    )
    return table, panel, DISCLAIMER


def _render(item: Estimate) -> None:
    table, panel, disclaimer = build_check_renderables(item)
    console.print(table)
    console.print(panel)
    console.print(f"[dim]{disclaimer}[/dim]")


def _evaluate(
    model: str,
    memory_gb: float | None,
    context: int,
    concurrency: int,
    kv_dtype: str,
    fixture: Path | None,
) -> Estimate:
    if context <= 0 or concurrency <= 0:
        raise typer.BadParameter("context and concurrency must be greater than zero")
    if kv_dtype not in DTYPE_BYTES:
        raise typer.BadParameter(f"kv-dtype must be one of: {', '.join(DTYPE_BYTES)}")
    try:
        info = resolve_model(model, fixture)
        return estimate(info, _memory(memory_gb), context, concurrency, kv_dtype)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    except Exception as exc:
        raise typer.BadParameter(f"could not read model metadata: {exc}") from exc


@app.command()
def system(
    json_output: Annotated[bool, typer.Option("--json", help="Machine-readable output")] = False,
) -> None:
    """Show Apple Silicon and unified-memory information."""
    info = detect_system()
    if json_output:
        typer.echo(json.dumps(info.to_dict(), indent=2))
        return
    table = Table(title="Mac AI Doctor · System", show_header=False)
    for key, value in (
        ("Platform", info.platform),
        ("Chip", info.chip or "unknown"),
        ("Model", info.model or "unknown"),
        ("Unified memory", f"{info.memory_gb:.1f} GB" if info.memory_gb else "not detected"),
    ):
        table.add_row(key, value)
    console.print(table)
    console.print(info.message)


@app.command()
def check(
    model: str,
    memory_gb: Annotated[float | None, typer.Option()] = None,
    context: Annotated[int, typer.Option()] = 4096,
    concurrency: Annotated[int, typer.Option()] = 1,
    kv_dtype: Annotated[str, typer.Option()] = "fp16",
    json_output: Annotated[bool, typer.Option("--json")] = False,
    fixture: Annotated[Path | None, typer.Option(hidden=True)] = None,
) -> None:
    """Estimate whether MODEL (HF ID, GGUF, or MLX/local config) fits."""
    result = _evaluate(model, memory_gb, context, concurrency, kv_dtype, fixture)
    typer.echo(json.dumps(result.to_dict(), indent=2)) if json_output else _render(result)


@app.command()
def compare(
    models: Annotated[list[str], typer.Argument(min=2)],
    memory_gb: Annotated[float | None, typer.Option()] = None,
    context: Annotated[int, typer.Option()] = 4096,
    concurrency: Annotated[int, typer.Option()] = 1,
    kv_dtype: Annotated[str, typer.Option()] = "fp16",
    json_output: Annotated[bool, typer.Option("--json")] = False,
    fixture: Annotated[Path | None, typer.Option(hidden=True)] = None,
) -> None:
    """Compare two or more models under the same workload."""
    results = [_evaluate(m, memory_gb, context, concurrency, kv_dtype, fixture) for m in models]
    if json_output:
        typer.echo(
            json.dumps(
                {"schema_version": "1.0", "results": [r.to_dict() for r in results]}, indent=2
            )
        )
        return
    table = Table(title="Model fit comparison")
    for name in ("Model", "Weights", "KV", "Peak", "Headroom", "Verdict"):
        table.add_column(name)

    def fmt(value: float | None) -> str:
        return f"{value:.2f} GB" if value is not None else "unknown"

    for result in results:
        table.add_row(
            result.model.model_id,
            fmt(result.weights_gb),
            fmt(result.kv_cache_gb),
            fmt(result.high_gb),
            fmt(result.headroom_gb),
            result.verdict.value,
        )
    console.print(table)


@app.command()
def recommend(
    memory_gb: Annotated[float, typer.Option(min=1)],
    task: Annotated[str, typer.Option()] = "chat",
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Suggest a conservative model class for coding, chat, or vision."""
    if task not in {"coding", "chat", "vision"}:
        raise typer.BadParameter("task must be coding, chat, or vision")
    budget = memory_gb * 0.65
    size = (
        3 if budget < 5 else 7 if budget < 10 else 14 if budget < 20 else 32 if budget < 40 else 70
    )
    quant = "4-bit"
    note = (
        f"Start with a {size}B-class {quant} {task} model. This leaves room for KV cache, runtime, "
        "macOS and other apps. Run `maid check` on the exact repository before downloading weights."
    )
    payload = {
        "schema_version": "1.0",
        "memory_gb": memory_gb,
        "task": task,
        "suggested_parameter_class_b": size,
        "quantization": quant,
        "guidance": note,
    }
    typer.echo(json.dumps(payload, indent=2)) if json_output else console.print(
        Panel(note, title="Conservative starting point")
    )


@app.command()
def web() -> None:
    """Launch the interactive Streamlit web UI for exploring model fit."""
    import importlib.util
    import subprocess
    import sys
    from pathlib import Path

    if importlib.util.find_spec("streamlit") is None:
        raise typer.BadParameter(
            "Streamlit is not installed. Install it with: uv tool install 'mac-ai-doctor[web]'"
        )
    app_path = Path(__file__).with_name("webapp.py")
    # --server.headless skips Streamlit's first-run email prompt and the browser
    # auto-open, which would otherwise block or misbehave when launched this way.
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.headless=true"]
    raise typer.Exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    app()
