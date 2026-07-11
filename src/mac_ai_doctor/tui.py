from __future__ import annotations

from dataclasses import dataclass

from rich.console import Group
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Footer, Header, Input, Label, Select, Static

from .cli import build_check_renderables
from .estimate import DTYPE_BYTES, estimate
from .metadata import resolve_model
from .models import Estimate
from .system import detect_system


@dataclass
class _Workload:
    """Validated inputs for a single estimate run."""

    model: str
    memory_gb: float
    context: int
    concurrency: int
    kv_dtype: str


def _parse(
    model: str, memory_raw: str, context_raw: str, concurrency_raw: str, kv_dtype: str
) -> _Workload:
    """Validate raw field text the same way cli._evaluate/_memory do; raise ValueError."""
    if not model.strip():
        raise ValueError("enter a model reference (HF ID, GGUF path, or local config dir)")
    if not memory_raw.strip():
        raise ValueError("enter available memory in GB")
    try:
        memory_gb = float(memory_raw)
    except ValueError:
        raise ValueError("memory must be a number") from None
    if memory_gb <= 0:
        raise ValueError("memory must be greater than zero")
    try:
        context = int(context_raw)
        concurrency = int(concurrency_raw)
    except ValueError:
        raise ValueError("context and concurrency must be whole numbers") from None
    if context <= 0 or concurrency <= 0:
        raise ValueError("context and concurrency must be greater than zero")
    if kv_dtype not in DTYPE_BYTES:
        raise ValueError(f"kv-dtype must be one of: {', '.join(DTYPE_BYTES)}")
    return _Workload(model.strip(), memory_gb, context, concurrency, kv_dtype)


class MaidApp(App[None]):
    """Interactive terminal UI mirroring `maid check`."""

    TITLE = "Mac AI Doctor"
    SUB_TITLE = "Will this model fit in your unified memory?"
    CSS = """
    #form { height: auto; padding: 1 2; }
    #form Label { margin-top: 1; }
    #controls { height: auto; margin-top: 1; }
    #controls Button { margin-right: 2; }
    #results { padding: 1 2; }
    .error { color: $error; }
    .hint { color: $text-muted; }
    """
    BINDINGS = [("q", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Header()
        detected = detect_system()
        memory_default = f"{detected.memory_gb:.0f}" if detected.memory_gb else ""
        with Vertical(id="form"):
            yield Label("Model reference (HF ID · GGUF path · local config dir)")
            yield Input(placeholder="ibm-granite/granite-3.1-8b-instruct", id="model")
            with Horizontal():
                with Vertical():
                    yield Label("Memory (GB)")
                    yield Input(value=memory_default, placeholder="16", id="memory")
                with Vertical():
                    yield Label("Context")
                    yield Input(value="4096", id="context")
                with Vertical():
                    yield Label("Concurrency")
                    yield Input(value="1", id="concurrency")
                with Vertical():
                    yield Label("KV dtype")
                    yield Select(
                        [(name, name) for name in DTYPE_BYTES],
                        value="fp16",
                        allow_blank=False,
                        id="kv_dtype",
                    )
            with Horizontal(id="controls"):
                yield Button("Check", variant="primary", id="check")
        with VerticalScroll(id="results"):
            yield Static(
                Text("Enter a model and press Check (or Enter). Weights are never downloaded."),
                id="output",
            )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#model", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "check":
            self._run_check()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._run_check()

    def _output(self) -> Static:
        return self.query_one("#output", Static)

    def _button(self) -> Button:
        return self.query_one("#check", Button)

    def _set_running(self, running: bool) -> None:
        button = self._button()
        button.disabled = running
        button.label = "Checking…" if running else "Check"

    def _run_check(self) -> None:
        if self._button().disabled:  # a check is already in flight
            return
        try:
            workload = _parse(
                self.query_one("#model", Input).value,
                self.query_one("#memory", Input).value,
                self.query_one("#context", Input).value,
                self.query_one("#concurrency", Input).value,
                str(self.query_one("#kv_dtype", Select).value),
            )
        except ValueError as exc:
            self._output().update(Text(str(exc), style="bold red"))
            return
        self._set_running(True)
        self._output().update(Text(f"Resolving {workload.model}…", style="dim"))
        self._estimate(workload)

    @work(thread=True, exclusive=True)
    def _estimate(self, workload: _Workload) -> None:
        try:
            info = resolve_model(workload.model)
            result = estimate(
                info,
                workload.memory_gb,
                workload.context,
                workload.concurrency,
                workload.kv_dtype,
            )
        except Exception as exc:  # noqa: BLE001 — surface any resolve/estimate failure inline
            message = Text(f"Could not read model metadata: {exc}", style="bold red")
            self.call_from_thread(self._finish, message)
            return
        self.call_from_thread(self._finish, result)

    def _finish(self, result: Estimate | Text) -> None:
        self._set_running(False)
        if isinstance(result, Text):
            self._output().update(result)
            return
        table, panel, disclaimer = build_check_renderables(result)
        self._output().update(Group(table, panel, Text(disclaimer, style="dim")))


def run_tui() -> None:
    """Launch the interactive terminal UI."""
    MaidApp().run()
