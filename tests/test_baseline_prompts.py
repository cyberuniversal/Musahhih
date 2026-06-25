import unittest

from scripts.prepare_nahw_eval import PROMPT as B0_TEMPLATE
from scripts.baseline_prompts import (
    B1_PROTOCOL_ID,
    B2_PROTOCOL_ID,
    PromptDemo,
    prompt_sha256,
    render_b0_prompt,
    render_b1_prompt,
    render_b2_prompt,
)


class PromptAssemblyTests(unittest.TestCase):
    def make_demos(self):
        return [
            PromptDemo(
                passage=f"نص المثال {index}",
                error=f"خطأ{index}",
                correction=f"صواب{index}",
            )
            for index in range(1, 6)
        ]

    def test_b0_renderer_uses_existing_frozen_template(self):
        self.assertEqual(
            render_b0_prompt("النص الأساسي", "خطأ"),
            B0_TEMPLATE.format(passage="النص الأساسي", error="خطأ"),
        )

    def test_b1_prompt_matches_frozen_template_exactly(self):
        expected = """فيما يلي خمسة أمثلة على المهمة نفسها. في كل مثال، أعدت الأداة الكلمة المصححة فقط.

المثال 1:
النص:
نص المثال 1
الكلمة الخاطئة:
خطأ1
الكلمة المصححة:
صواب1

المثال 2:
النص:
نص المثال 2
الكلمة الخاطئة:
خطأ2
الكلمة المصححة:
صواب2

المثال 3:
النص:
نص المثال 3
الكلمة الخاطئة:
خطأ3
الكلمة المصححة:
صواب3

المثال 4:
النص:
نص المثال 4
الكلمة الخاطئة:
خطأ4
الكلمة المصححة:
صواب4

المثال 5:
النص:
نص المثال 5
الكلمة الخاطئة:
خطأ5
الكلمة المصححة:
صواب5

الآن نفذ المهمة على النص التالي.
صحح الكلمة الخاطئة المحددة في النص التالي.
أعد الكلمة المصححة فقط دون شرح أو علامات اقتباس.

النص:
نص الاختبار

الكلمة الخاطئة:
خطأ الاختبار
"""
        self.assertEqual(
            render_b1_prompt(self.make_demos(), "نص الاختبار", "خطأ الاختبار"),
            expected,
        )

    def test_b1_requires_exactly_five_demos(self):
        with self.assertRaisesRegex(ValueError, "exactly five"):
            render_b1_prompt(self.make_demos()[:4], "نص", "خطأ")
        with self.assertRaisesRegex(ValueError, "exactly five"):
            render_b1_prompt(self.make_demos() + self.make_demos()[:1], "نص", "خطأ")

    def test_b2_prompt_matches_frozen_template_exactly(self):
        expected = """أنت أداة متخصصة في تصحيح العربية الفصحى.
راجع سياق النص لتحديد الصيغة الصحيحة للكلمة المحددة، مع إبقاء بقية النص دون تغيير.
أعد الكلمة المصححة فقط دون شرح أو تعليل أو علامات اقتباس.

النص:
نص الاختبار

الكلمة الخاطئة:
خطأ الاختبار
"""
        self.assertEqual(render_b2_prompt("نص الاختبار", "خطأ الاختبار"), expected)

    def test_protocol_ids_and_prompt_hash_are_stable(self):
        self.assertEqual(B1_PROTOCOL_ID, "B1-P1")
        self.assertEqual(B2_PROTOCOL_ID, "B2-P1")
        self.assertEqual(
            prompt_sha256("abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        )


if __name__ == "__main__":
    unittest.main()
