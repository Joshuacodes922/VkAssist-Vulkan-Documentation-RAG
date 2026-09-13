from dataclasses import dataclass
import asyncio
import json
import re

import httpx

from app.core.settings import settings
from app.retrieval.hybrid import HybridResult


class AnsweringNotConfigured(RuntimeError):
    pass


class InvalidGeneratedAnswer(RuntimeError):
    pass


@dataclass(frozen=True)
class GroundedAnswer:
    answer: str
    citations: list[dict[str, object]]


_generation_lock = asyncio.Lock()


def clean_answer(text: str) -> str:
    """Do not expose provider reasoning traces in the user-facing answer."""
    cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and isinstance(parsed.get("answer"), str):
            return parsed["answer"].strip()
    except json.JSONDecodeError:
        pass

    # Some thinking models prepend internal text before the JSON object.
    matches = re.findall(r"\{\s*\"answer\"\s*:\s*\"(?:[^\"\\]|\\.)*\"\s*\}", cleaned, re.DOTALL)
    if matches:
        return json.loads(matches[-1])["answer"].strip()
    return cleaned


def is_placeholder_answer(answer: str) -> bool:
    return answer.strip().strip(".") == ""


def build_context(results: list[HybridResult]) -> tuple[str, list[dict[str, object]]]:
    passages: list[str] = []
    citations: list[dict[str, object]] = []
    for index, result in enumerate(results, start=1):
        chunk = result.chunk
        citation = {
            "id": index,
            "document": chunk.document_name,
            "page_start": chunk.page_number,
            "page_end": chunk.page_end,
            "section": chunk.section,
        }
        citations.append(citation)
        passages.append(f"[{index}] {chunk.content}")
    return "\n\n".join(passages), citations


async def answer_with_citations(query: str, results: list[HybridResult]) -> GroundedAnswer:
    if not (settings.llm_base_url and settings.llm_model):
        raise AnsweringNotConfigured(
            "Set LLM_BASE_URL and LLM_MODEL to enable answer generation."
        )

    context, citations = build_context(results)
    prompt = (
        "Answer only from the supplied Vulkan documentation passages. "
        "Use at most three concise sentences. Cite every factual claim using "
        "[n]. If the passages do not support an answer, say so.\n\n"
        f"Question: {query}\n\nPassages:\n{context}"
    )
    headers = {"Content-Type": "application/json"}
    if settings.llm_api_key:
        headers["Authorization"] = f"Bearer {settings.llm_api_key}"
    messages = [
        {
            "role": "system",
            "content": (
                "Return exactly one JSON object with one key named answer. "
                "Its value must be the complete final natural-language answer, "
                "not a template, placeholder, analysis, or markdown. Never use "
                "an ellipsis as the answer."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    payload = {
        "model": settings.llm_model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": settings.llm_max_tokens,
        "response_format": {"type": "json_object"},
        # Qwen thinking is useful for difficult tasks but adds substantial
        # latency to grounded documentation answers.
        "reasoning_effort": "none",
    }
    async with _generation_lock:
        async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
            for attempt in range(2):
                response = await client.post(
                    f"{settings.llm_base_url.rstrip('/')}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                answer = clean_answer(
                    response.json()["choices"][0]["message"]["content"]
                )
                if not is_placeholder_answer(answer):
                    cited_ids = {int(value) for value in re.findall(r"\[(\d+)\]", answer)}
                    return GroundedAnswer(
                        answer=answer,
                        citations=[citation for citation in citations if citation["id"] in cited_ids],
                    )
                if attempt == 0:
                    payload["messages"] = messages + [
                        {
                            "role": "user",
                            "content": "Your previous output was a placeholder. Produce the complete answer now.",
                        }
                    ]

    raise InvalidGeneratedAnswer("The model returned a placeholder instead of an answer.")
