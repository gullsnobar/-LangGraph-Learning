"""Deterministic tests for LangGraph state, routing, and retry behavior."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import app
from retry_workflow import (
    AnalysisResult,
    ImprovementResult,
    QualityReview,
    WorkflowInputError,
    create_initial_state,
    create_workflow,
    route_review,
    run_product_review,
)


class ScriptedReviewer:
    """Test double that controls scores while exercising the real graph."""

    def __init__(self, scores: list[int]) -> None:
        self.scores = iter(scores)
        self.analyze_calls = 0
        self.improve_calls = 0
        self.evaluate_calls = 0

    def analyze(self, description: str) -> AnalysisResult:
        self.analyze_calls += 1
        return AnalysisResult(
            summary=f"Analysis of {description}",
            strengths=["Contains a named product."],
            improvement_opportunities=["Connect features to customer benefits."],
        )

    def improve(
        self,
        original_description: str,
        current_description: str,
        analysis: str,
        review_feedback: str,
        attempt: int,
    ) -> ImprovementResult:
        self.improve_calls += 1
        return ImprovementResult(
            description=f"{current_description} Improved copy."
        )

    def evaluate(
        self,
        original_description: str,
        improved_description: str,
    ) -> QualityReview:
        self.evaluate_calls += 1
        score = next(self.scores)
        return QualityReview(
            score=score,
            rationale=f"The scripted score is {score}.",
            remaining_issues=[] if score >= 8 else ["More refinement is required."],
            unsupported_claims=[],
        )


class WorkflowTests(unittest.TestCase):
    def test_graph_contains_required_nodes(self) -> None:
        workflow = create_workflow(ScriptedReviewer([8]))
        node_names = set(workflow.get_graph().nodes)

        self.assertTrue(
            {"analyze", "improve", "review", "approved"}.issubset(node_names)
        )

    def test_first_review_can_route_to_approval(self) -> None:
        reviewer = ScriptedReviewer([8])
        workflow = create_workflow(reviewer)

        result = run_product_review("A factual product description.", 5, workflow)

        self.assertEqual(result["status"], "approved")
        self.assertEqual(result["quality_score"], 8)
        self.assertEqual(result["retry_count"], 1)
        self.assertEqual(reviewer.analyze_calls, 1)
        self.assertEqual(reviewer.improve_calls, 1)
        self.assertEqual(reviewer.evaluate_calls, 1)

    def test_conditional_edge_retries_until_content_is_approved(self) -> None:
        reviewer = ScriptedReviewer([5, 7, 9])
        workflow = create_workflow(reviewer)

        result = run_product_review("A factual product description.", 4, workflow)

        self.assertEqual(result["status"], "approved")
        self.assertEqual(result["quality_score"], 9)
        self.assertEqual(result["retry_count"], 3)
        self.assertEqual(reviewer.improve_calls, 3)
        self.assertIn("conditional route = retry", " ".join(result["trace"]))
        self.assertIn("conditional route = approve", " ".join(result["trace"]))

    def test_retry_limit_stops_cycle_and_routes_to_manual_review(self) -> None:
        reviewer = ScriptedReviewer([4, 6, 9])
        workflow = create_workflow(reviewer)

        result = run_product_review("A factual product description.", 2, workflow)

        self.assertEqual(result["status"], "needs_manual_review")
        self.assertEqual(result["quality_score"], 6)
        self.assertEqual(result["retry_count"], 2)
        self.assertEqual(reviewer.improve_calls, 2)
        self.assertEqual(reviewer.evaluate_calls, 2)
        self.assertIn("retry limit reached", result["trace"][-1].lower())

    def test_numeric_factuality_guard_prevents_false_approval(self) -> None:
        class HallucinatingReviewer(ScriptedReviewer):
            def improve(
                self,
                original_description: str,
                current_description: str,
                analysis: str,
                review_feedback: str,
                attempt: int,
            ) -> ImprovementResult:
                self.improve_calls += 1
                return ImprovementResult(
                    description="The battery provides 12 hours of use."
                )

        reviewer = HallucinatingReviewer([9])
        workflow = create_workflow(reviewer)

        result = run_product_review("The battery lasts through a workday.", 1, workflow)

        self.assertEqual(result["quality_score"], 5)
        self.assertEqual(result["status"], "needs_manual_review")
        self.assertIn("numeric claim '12'", result["review_feedback"])

    def test_routing_function_covers_all_destinations(self) -> None:
        state = create_initial_state("Product description", 3)

        state["quality_score"] = 8
        self.assertEqual(route_review(state), "approved")

        state["quality_score"] = 7
        state["retry_count"] = 1
        self.assertEqual(route_review(state), "retry")

        state["retry_count"] = 3
        self.assertEqual(route_review(state), "max_retries_reached")

    def test_invalid_description_is_rejected_before_graph_execution(self) -> None:
        for invalid in ("", "   ", None, "x" * 5_001):
            with self.subTest(invalid=invalid if invalid is None else "text"):
                with self.assertRaises(WorkflowInputError):
                    create_initial_state(invalid, 3)

    def test_invalid_attempt_limits_are_rejected(self) -> None:
        for invalid in (0, 6, 1.5, "many", None, True):
            with self.subTest(invalid=invalid):
                with self.assertRaises(WorkflowInputError):
                    create_initial_state("Product description", invalid)


class AppTests(unittest.TestCase):
    def test_empty_input_returns_friendly_message_without_starting_graph(self) -> None:
        outputs = app.review_product("  ", 3)

        self.assertIn("Enter a product description", outputs[0])
        self.assertIsNone(outputs[3])
        self.assertEqual(outputs[4], 0)
        self.assertIn("Input needed", outputs[5])

    def test_successful_state_is_mapped_to_every_ui_output(self) -> None:
        result = {
            "analysis": "Useful analysis",
            "improved_description": "Improved copy",
            "review_feedback": "Review feedback",
            "quality_score": 9,
            "retry_count": 2,
            "status": "approved",
            "trace": ["Analyze", "Improve", "Review"],
        }
        with patch.object(app, "run_product_review", return_value=result):
            outputs = app.review_product("Product copy", 3)

        self.assertEqual(outputs[0:5], ("Useful analysis", "Improved copy", "Review feedback", 9, 2))
        self.assertIn("Approved", outputs[5])
        self.assertIn("1. Analyze", outputs[6])

    def test_runtime_error_is_reported_without_crashing_interface(self) -> None:
        with (
            patch.object(
                app,
                "run_product_review",
                side_effect=ConnectionError("Connection refused by Ollama"),
            ),
            patch.object(app.LOGGER, "exception"),
        ):
            outputs = app.review_product("Product copy", 3)

        self.assertEqual(outputs[0], "The review could not be completed.")
        self.assertIn("Could not connect to Ollama", outputs[2])
        self.assertIn("Review failed", outputs[5])


if __name__ == "__main__":
    unittest.main()
