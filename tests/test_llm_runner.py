from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from redllm import DATASET_SOURCES, LLMRunner
from redllm import llm_runner as runner_module
from redllm.cli import main, parse_pipeline_options


class DummyPipeline:
    def __init__(self, suffix: str = "::response::"):
        self.suffix = suffix
        self.calls = []

    def __call__(self, prompt: str, **kwargs):
        self.calls.append((prompt, kwargs))
        return [{"generated_text": f"{prompt} {self.suffix}"}]


def test_generate_uses_pipeline(monkeypatch):
    pipeline = DummyPipeline()
    monkeypatch.setattr(runner_module, "_load_text_generation_pipeline", lambda name, **_: pipeline)

    runner = LLMRunner.from_pretrained(model_name="custom-model")
    outputs = runner.generate("Hello")

    assert outputs == ["Hello ::response::"]
    assert pipeline.calls == [("Hello", {"max_new_tokens": 128})]


def test_batched_generate(monkeypatch):
    pipeline = DummyPipeline("::completion::")
    monkeypatch.setattr(runner_module, "_load_text_generation_pipeline", lambda name, **_: pipeline)

    runner = LLMRunner.from_pretrained()
    outputs = runner.batched_generate(["Hi", "Bye"], max_new_tokens=3)

    assert outputs == [["Hi ::completion::"], ["Bye ::completion::"]]
    assert pipeline.calls == [
        ("Hi", {"max_new_tokens": 3}),
        ("Bye", {"max_new_tokens": 3}),
    ]


def test_dataset_description_mentions_sources():
    description = LLMRunner.dataset_description()
    for source in DATASET_SOURCES:
        assert source in description


def test_cli_without_arguments_prints_help(capsys):
    rc = main([])
    captured = capsys.readouterr()

    assert rc == 0
    assert "usage:" in captured.out


def test_cli_prompts_generate_output(monkeypatch, capsys):
    pipeline = DummyPipeline()
    monkeypatch.setattr(runner_module, "_load_text_generation_pipeline", lambda name, **_: pipeline)

    rc = main(["--prompt", "Test prompt", "--max-new-tokens", "5"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Prompt: Test prompt" in captured.out
    assert "Response 1: Test prompt ::response::" in captured.out
    assert pipeline.calls == [("Test prompt", {"max_new_tokens": 5})]


def test_cli_chat_mode(monkeypatch, capsys):
    pipeline = DummyPipeline()
    monkeypatch.setattr(runner_module, "_load_text_generation_pipeline", lambda name, **_: pipeline)

    prompts = iter(["Hello there"])
    seen_prompts = []

    def fake_input(prompt: str) -> str:
        seen_prompts.append(prompt)
        try:
            return next(prompts)
        except StopIteration:
            raise EOFError

    rc = main(["--chat"], input_func=fake_input)
    captured = capsys.readouterr()

    assert rc == 0
    assert seen_prompts == ["You: ", "You: "]
    assert "Starting interactive chat." in captured.out
    assert "Model: Hello there ::response::" in captured.out
    assert "Exiting chat." in captured.out
    assert pipeline.calls == [("Hello there", {"max_new_tokens": 128})]


@pytest.mark.parametrize(
    "options, expected",
    [
        ([], {}),
        (["temperature=0.9"], {"temperature": "0.9"}),
        (["top_k=50", "top_p=0.95"], {"top_k": "50", "top_p": "0.95"}),
    ],
)
def test_parse_pipeline_options_success(options, expected):
    assert parse_pipeline_options(options) == expected


def test_parse_pipeline_options_error():
    with pytest.raises(ValueError):
        parse_pipeline_options(["invalid"])


def test_redllm_script_bootstraps_package(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(repo_root / "redllm")],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout


def test_start_llm_wrapper_bootstraps_package(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "start_llm.py")],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout
