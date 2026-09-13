import unittest

from app.evaluation import metrics_for_ranking
from app.services.answering import clean_answer, is_placeholder_answer


class Chunk:
    def __init__(self, page_number: int, page_end: int) -> None:
        self.page_number, self.page_end = page_number, page_end


class Result:
    def __init__(self, start: int, end: int) -> None:
        self.chunk = Chunk(start, end)


class EvaluationTests(unittest.TestCase):
    def test_metrics_reward_a_cross_page_relevant_chunk(self) -> None:
        metrics = metrics_for_ranking([Result(10, 11), Result(20, 20)], frozenset({11}), 2)
        self.assertEqual(metrics, {"recall": 1.0, "reciprocal_rank": 1.0, "ndcg": 1.0})

    def test_metrics_do_not_reward_duplicate_relevant_pages(self) -> None:
        metrics = metrics_for_ranking(
            [Result(10, 10), Result(10, 10), Result(20, 20)],
            frozenset({10}),
            3,
        )
        self.assertEqual(metrics, {"recall": 1.0, "reciprocal_rank": 1.0, "ndcg": 1.0})

    def test_removes_provider_reasoning_from_an_answer(self) -> None:
        self.assertEqual(clean_answer("<think>private reasoning</think>Answer [1]"), "Answer [1]")

    def test_extracts_json_answer_after_provider_reasoning(self) -> None:
        raw = 'analysis\n{"answer":"A swap chain presents images [1]."}'
        self.assertEqual(clean_answer(raw), "A swap chain presents images [1].")

    def test_recognizes_empty_and_ellipsis_answers_as_placeholders(self) -> None:
        self.assertTrue(is_placeholder_answer("..."))
        self.assertTrue(is_placeholder_answer("  "))
        self.assertFalse(is_placeholder_answer("A swap chain presents images [1]."))


if __name__ == "__main__":
    unittest.main()
