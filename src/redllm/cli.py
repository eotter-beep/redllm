"""Command-line utilities for interacting with the GPT-2 language model runner."""
from __future__ import annotations

import argparse
from typing import Callable, Sequence

from . import DEFAULT_MODEL_NAME, LLMRunner


def build_parser() -> argparse.ArgumentParser:
    """Create an argument parser configured for the redllm CLI."""
    parser = argparse.ArgumentParser(
        prog="redllm",
        description="Interact with the GPT-2 based language model runner.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="Name or path of the pretrained model to load (default: %(default)s).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate per prompt (default: %(default)s).",
    )
    parser.add_argument(
        "--pipeline-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Extra keyword arguments forwarded to the transformers pipeline. "
            "Specify as KEY=VALUE; the value is parsed as a string."
        ),
    )
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        default=[],
        help="Prompt to send to the model in batch mode. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Start an interactive chat session with the model.",
    )
    return parser


def parse_pipeline_options(values: Sequence[str]) -> dict[str, str]:
    """Convert KEY=VALUE strings into a dictionary of pipeline options."""
    options: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid pipeline option '{item}'. Expected KEY=VALUE format.")
        key, value = item.split("=", 1)
        options[key] = value
    return options


def run_prompts(
    prompts: Sequence[str],
    *,
    model_name: str,
    max_new_tokens: int,
    output_func: Callable[[str], None] = print,
    **pipeline_options,
) -> int:
    """Generate responses for a collection of prompts."""
    runner = LLMRunner.from_pretrained(model_name, **pipeline_options)
    if not prompts:
        output_func(LLMRunner.dataset_description())
        return 0

    for prompt in prompts:
        output_func(f"\nPrompt: {prompt}")
        responses = runner.generate(prompt, max_new_tokens=max_new_tokens)
        for index, text in enumerate(responses, start=1):
            output_func(f"Response {index}: {text}")
    return 0


def run_chat(
    *,
    model_name: str,
    max_new_tokens: int,
    input_func: Callable[[str], str] = input,
    output_func: Callable[[str], None] = print,
    **pipeline_options,
) -> int:
    """Start an interactive chat session with the language model."""
    runner = LLMRunner.from_pretrained(model_name, **pipeline_options)
    output_func("Starting interactive chat. Press Ctrl-D to exit.")

    while True:
        try:
            prompt = input_func("You: ")
        except (EOFError, KeyboardInterrupt):
            output_func("Exiting chat.")
            return 0

        if not prompt.strip():
            continue

        responses = runner.generate(prompt, max_new_tokens=max_new_tokens)
        for text in responses:
            output_func(f"Model: {text}")


def main(
    argv: Sequence[str] | None = None,
    *,
    input_func: Callable[[str], str] = input,
    output_func: Callable[[str], None] = print,
) -> int:
    """Entry point used by both the redllm CLI script and tests."""
    parser = build_parser()

    if argv is None:
        import sys

        argv = sys.argv[1:]

    if not list(argv):
        parser.print_help()
        return 0

    args = parser.parse_args(argv)

    if args.chat and args.prompts:
        parser.error("--chat cannot be combined with --prompt")

    pipeline_options = parse_pipeline_options(args.pipeline_option)

    if args.chat:
        return run_chat(
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
            input_func=input_func,
            output_func=output_func,
            **pipeline_options,
        )

    return run_prompts(
        args.prompts,
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        output_func=output_func,
        **pipeline_options,
    )


__all__ = [
    "build_parser",
    "parse_pipeline_options",
    "run_prompts",
    "run_chat",
    "main",
]
