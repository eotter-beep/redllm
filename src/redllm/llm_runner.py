"""Tools for starting and interacting with GPT-2 based language models."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Sequence

DEFAULT_MODEL_NAME = "gpt2"
DATASET_SOURCES: Sequence[str] = (
    "a-m-team/AM-DeepSeek-R1-Distilled-1.4M",
    "nvidia/Llama-Nemotron-Post-Training-Dataset",
    "Liontix/claude-sonnet-4-100x",
)


def _import_transformers_pipeline() -> Callable:
    """Return the :func:`transformers.pipeline` callable.

    The import is performed lazily so that unit tests can substitute
    the dependency and so that users receive a descriptive error when the
    optional dependency is unavailable.
    """

    try:
        from transformers import pipeline  # type: ignore import-not-found
    except ImportError as exc:  # pragma: no cover - defensive branch
        raise RuntimeError(
            "The `transformers` package is required to load the language model."
        ) from exc
    return pipeline


def _load_text_generation_pipeline(model_name: str, **pipeline_kwargs):
    """Create a text-generation pipeline for the given model name."""

    pipeline_fn = _import_transformers_pipeline()
    default_kwargs = {"model": model_name, "tokenizer": model_name}
    default_kwargs.update(pipeline_kwargs)
    return pipeline_fn("text-generation", **default_kwargs)


@dataclass
class LLMRunner:
    """Utility wrapper for a GPT-2 based language model."""

    model_name: str = DEFAULT_MODEL_NAME
    _pipeline_factory: Callable[[str], Callable[[str], Iterable[dict]]] = field(
        default=_load_text_generation_pipeline, repr=False
    )
    _pipeline: Callable[[str, int], List[dict]] | None = field(default=None, init=False, repr=False)

    def _ensure_pipeline(self) -> Callable[[str, int], List[dict]]:
        if self._pipeline is None:
            self._pipeline = self._pipeline_factory(self.model_name)
        return self._pipeline

    def generate(self, prompt: str, *, max_new_tokens: int = 128, **generation_kwargs) -> List[str]:
        """Generate completions for a single prompt."""

        pipeline = self._ensure_pipeline()
        responses = pipeline(prompt, max_new_tokens=max_new_tokens, **generation_kwargs)
        return [entry.get("generated_text", "") for entry in responses]

    def batched_generate(
        self, prompts: Sequence[str], *, max_new_tokens: int = 128, **generation_kwargs
    ) -> List[List[str]]:
        """Generate completions for multiple prompts."""

        pipeline = self._ensure_pipeline()
        outputs: List[List[str]] = []
        for prompt in prompts:
            responses = pipeline(prompt, max_new_tokens=max_new_tokens, **generation_kwargs)
            outputs.append([entry.get("generated_text", "") for entry in responses])
        return outputs

    @classmethod
    def from_pretrained(cls, model_name: str = DEFAULT_MODEL_NAME, **pipeline_kwargs) -> "LLMRunner":
        """Instantiate a runner with a freshly loaded transformers pipeline."""

        def factory(name: str) -> Callable[[str, int], List[dict]]:
            return _load_text_generation_pipeline(name, **pipeline_kwargs)

        return cls(model_name=model_name, _pipeline_factory=factory)

    @staticmethod
    def dataset_description() -> str:
        """Return a human readable description of the training sources."""

        return (
            "This runner targets a GPT-2 based model fine-tuned with the following "
            f"datasets: {', '.join(DATASET_SOURCES)}."
        )


__all__ = ["LLMRunner", "DATASET_SOURCES", "DEFAULT_MODEL_NAME"]
