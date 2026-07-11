import json
import struct
from pathlib import Path

from typer.testing import CliRunner

from mac_ai_doctor.cli import app, build_check_renderables
from mac_ai_doctor.estimate import estimate
from mac_ai_doctor.metadata import _normalize_hf_id, resolve_fixture, resolve_gguf
from mac_ai_doctor.models import Verdict

FIXTURE = "tests/fixtures/llama-8b.json"


def test_fixture_and_estimator():
    from pathlib import Path

    info = resolve_fixture(Path(FIXTURE))
    result = estimate(info, 16, context=4096)
    assert result.verdict == Verdict.COMFORTABLE
    assert result.kv_cache_gb and result.kv_cache_gb > 0
    assert result.high_gb is not None and result.weights_gb is not None
    assert result.high_gb > result.weights_gb


def test_gguf_bounded_header(tmp_path):
    path = tmp_path / "tiny.gguf"
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 2, 4) + b"padding")
    assert resolve_gguf(path).weight_bytes == path.stat().st_size


def test_cli_check_json():
    result = CliRunner().invoke(
        app, ["check", "ignored", "--memory-gb", "16", "--fixture", FIXTURE, "--json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema_version"] == "1.0"
    assert payload["verdict"] == "COMFORTABLE"


def test_cli_system_linux_is_friendly():
    result = CliRunner().invoke(app, ["system", "--json"])
    assert result.exit_code == 0
    assert "supported" in json.loads(result.output)


def test_recommend_validation_and_json():
    runner = CliRunner()
    assert (
        runner.invoke(
            app, ["recommend", "--memory-gb", "16", "--task", "coding", "--json"]
        ).exit_code
        == 0
    )
    assert runner.invoke(app, ["recommend", "--memory-gb", "16", "--task", "music"]).exit_code != 0


def test_build_check_renderables_matches_estimate():
    info = resolve_fixture(Path(FIXTURE))
    result = estimate(info, 16, context=4096)
    table, panel, disclaimer = build_check_renderables(result)
    assert table.title == result.model.model_id
    assert result.verdict.value in str(panel.title)
    assert "benchmark" in disclaimer


def test_web_command_registered():
    assert CliRunner().invoke(app, ["web", "--help"]).exit_code == 0


def test_webapp_importable_and_reuses_estimate():
    # The Streamlit app must import cleanly (no Streamlit runtime needed) and
    # expose main() plus the shared verdict/DTYPE surfaces it renders from.
    from mac_ai_doctor import webapp

    assert callable(webapp.main)
    assert set(webapp._VERDICT_STYLE) == set(Verdict)


def test_normalize_hf_id_accepts_urls():
    ident = "ibm-granite/granite-switch-4.1-3b-preview"
    assert _normalize_hf_id(ident) == ident
    assert _normalize_hf_id(f"https://huggingface.co/{ident}") == ident
    assert _normalize_hf_id(f"  https://huggingface.co/{ident}  ") == ident
    assert _normalize_hf_id(f"huggingface.co/{ident}") == ident
    assert _normalize_hf_id(f"https://huggingface.co/{ident}/tree/main") == ident
    assert _normalize_hf_id(f"https://huggingface.co/{ident}?library=transformers") == ident
