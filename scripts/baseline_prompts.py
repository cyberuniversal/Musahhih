#!/usr/bin/env python3
"""Frozen prompt assembly helpers for Musahhih prompt-only baselines."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

from scripts.prepare_nahw_eval import PROMPT as B0_TEMPLATE


B0_PROTOCOL_ID = "B0-P1"
B1_PROTOCOL_ID = "B1-P1"
B2_PROTOCOL_ID = "B2-P1"
EXPECTED_B1_DEMO_COUNT = 5


@dataclass(frozen=True)
class PromptDemo:
    """A private B1 demonstration row.

    Values may contain licensed QALB text. Callers must keep instances and any
    rendered B1 prompt out of public logs, Git, issues, and PRs.
    """

    passage: str
    error: str
    correction: str


def prompt_sha256(prompt: str) -> str:
    """Return the SHA-256 hash of a prompt's exact UTF-8 bytes."""

    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def render_b0_prompt(passage: str, error: str) -> str:
    """Render the already accepted B0-P1 frozen prompt."""

    return B0_TEMPLATE.format(passage=passage, error=error)


def render_b1_prompt(
    demos: list[PromptDemo] | tuple[PromptDemo, ...],
    passage: str,
    error: str,
) -> str:
    """Render the frozen B1-P1 five-shot prompt exactly."""

    if len(demos) != EXPECTED_B1_DEMO_COUNT:
        raise ValueError("B1-P1 requires exactly five demonstrations")
    return f"""فيما يلي خمسة أمثلة على المهمة نفسها. في كل مثال، أعدت الأداة الكلمة المصححة فقط.

المثال 1:
النص:
{demos[0].passage}
الكلمة الخاطئة:
{demos[0].error}
الكلمة المصححة:
{demos[0].correction}

المثال 2:
النص:
{demos[1].passage}
الكلمة الخاطئة:
{demos[1].error}
الكلمة المصححة:
{demos[1].correction}

المثال 3:
النص:
{demos[2].passage}
الكلمة الخاطئة:
{demos[2].error}
الكلمة المصححة:
{demos[2].correction}

المثال 4:
النص:
{demos[3].passage}
الكلمة الخاطئة:
{demos[3].error}
الكلمة المصححة:
{demos[3].correction}

المثال 5:
النص:
{demos[4].passage}
الكلمة الخاطئة:
{demos[4].error}
الكلمة المصححة:
{demos[4].correction}

الآن نفذ المهمة على النص التالي.
صحح الكلمة الخاطئة المحددة في النص التالي.
أعد الكلمة المصححة فقط دون شرح أو علامات اقتباس.

النص:
{passage}

الكلمة الخاطئة:
{error}
"""


def render_b2_prompt(passage: str, error: str) -> str:
    """Render the frozen B2-P1 expert-style prompt exactly."""

    return f"""أنت أداة متخصصة في تصحيح العربية الفصحى.
راجع سياق النص لتحديد الصيغة الصحيحة للكلمة المحددة، مع إبقاء بقية النص دون تغيير.
أعد الكلمة المصححة فقط دون شرح أو تعليل أو علامات اقتباس.

النص:
{passage}

الكلمة الخاطئة:
{error}
"""
