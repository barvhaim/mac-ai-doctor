import json
import struct

from typer.testing import CliRunner

from mac_ai_doctor.cli import app
from mac_ai_doctor.estimate import estimate
from mac_ai_doctor.metadata import resolve_fixture, resolve_gguf
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
