"""LangGraph workflow for reviewing and improving product descriptions.

The production workflow uses a local Ollama model for every analysis,
improvement, and quality-review step. The reviewer dependency can be replaced
in tests so routing and retry behavior can be verified deterministically.
"""

from __future__ import annotations

import operator
import os
import re
from typing import Annotated, Literal, Protocol, TypedDict, cast

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field


APPROVAL_SCORE = 8
DEFAULT_MODEL = "llama3.2:latest"
MAX_DESCRIPTION_LENGTH = 5_000
MIN_RETRIES = 1
MAX_RETRIES = 5


class WorkflowInputError(ValueError):
    """Raised when user-supplied workflow input is invalid."""


class ProductState(TypedDict):
    """Shared state passed between every node in the graph."""

    product_description: str
    analysis: str
    improved_description: str
    review_feedback: str
    quality_score: int
    retry_count: int
    max_retries: int
    status: str
    trace: Annotated[list[str], operator.add]


class AnalysisResult(BaseModel):
    """Structured result produced by the analysis model call."""

    summary: str = Field(description="A concise assessment of the description")
    strengths: list[str] = Field(description="Specific strengths already present")
    improvement_opportunities: list[str] = Field(
        description="Specific, actionable ways to improve the description"
    )


class ImprovementResult(BaseModel):
    """Structured result produced by the improvement model call."""

    description: str = Field(
        description="The revised, publication-ready product description"
    )


class QualityReview(BaseModel):
    """Structured result produced by the quality-review model call."""

    score: int = Field(
        ge=1,
        le=10,
        description="Overall quality score from 1 (poor) to 10 (excellent)",
    )
    rationale: str = Field(description="A concise explanation of the score")
    remaining_issues: list[str] = Field(
        description="Actionable issues that still prevent a higher score"
    )
    unsupported_claims: list[str] = Field(
        description="Claims in the candidate that are not supported by the original"
    )


class ProductReviewer(Protocol):
    """Operations required by the graph nodes."""

    def analyze(self, description: str) -> AnalysisResult:
        """Analyze the original description."""

    def improve(
        self,
        original_description: str,
        current_description: str,
        analysis: str,
        review_feedback: str,
        attempt: int,
    ) -> ImprovementResult:
        """Improve a description using all available state."""

    def evaluate(
        self,
        original_description: str,
        improved_description: str,
    ) -> QualityReview:
        """Score an improved description."""


class OllamaProductReviewer:
    """Real reviewer implementation backed by a local Ollama chat model."""

    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
    ) -> None:
        resolved_model = model_name or os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)
        resolved_url = base_url or os.getenv(
            "OLLAMA_BASE_URL", "http://127.0.0.1:11434"
        )
        model = ChatOllama(
            model=resolved_model,
            base_url=resolved_url,
            temperature=0,
            num_predict=600,
        )

        analysis_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a meticulous ecommerce content strategist. Analyze "
                    "only the facts present in the supplied product description. "
                    "Never invent specifications, certifications, prices, or claims. "
                    "Separate improvements possible with current facts from missing "
                    "information the author would need to supply.",
                ),
                (
                    "human",
                    "Analyze this product description for clarity, specificity, "
                    "customer benefits, credibility, and completeness. Identify "
                    "what already works and give actionable improvements.\n\n"
                    "PRODUCT DESCRIPTION:\n{description}",
                ),
            ]
        )
        improvement_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You improve ecommerce product copy while preserving factual "
                    "accuracy. Never add a feature or claim that is not supported "
                    "by the original description. Copy every numerical fact exactly; "
                    "never convert qualitative wording into a number (for example, "
                    "'a full workday' must not become '12 hours'). If facts are "
                    "limited, produce shorter copy instead of padding it. Avoid "
                    "subjective or absolute phrases such as 'seamless', 'powerful', "
                    "'boost productivity', and 'long-lasting' unless the original "
                    "supports them. Return polished copy only through the requested "
                    "structured field.",
                ),
                (
                    "human",
                    "Rewrite the current product description so it is clear, "
                    "specific, benefit-focused, professional, and easy to scan. "
                    "Use one concise paragraph and only the facts available in the "
                    "original. This is improvement attempt {attempt}.\n\n"
                    "ORIGINAL DESCRIPTION (the factual source of truth):\n"
                    "{original_description}\n\n"
                    "CURRENT DESCRIPTION:\n{current_description}\n\n"
                    "INITIAL ANALYSIS:\n{analysis}\n\n"
                    "LATEST REVIEW FEEDBACK:\n{review_feedback}",
                ),
            ]
        )
        review_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a strict but fair ecommerce copy editor. Score the "
                    "candidate on clarity, specificity, customer benefits, "
                    "credibility, and completeness. Penalize invented or unsupported "
                    "claims. First compare every concrete and numerical claim with "
                    "the original. Any unsupported claim must be listed and limits "
                    "the score to 5. Judge completeness relative to the facts the "
                    "author supplied; do not demand missing specifications that the "
                    "writer cannot truthfully add. A score of 8 or higher means "
                    "publication-ready.",
                ),
                (
                    "human",
                    "Evaluate the candidate description against the original facts. "
                    "Explain the score and list any remaining actionable issues.\n\n"
                    "ORIGINAL DESCRIPTION:\n{original_description}\n\n"
                    "CANDIDATE DESCRIPTION:\n{improved_description}",
                ),
            ]
        )

        self._analysis_chain = analysis_prompt | model.with_structured_output(
            AnalysisResult
        )
        self._improvement_chain = improvement_prompt | model.with_structured_output(
            ImprovementResult
        )
        self._review_chain = review_prompt | model.with_structured_output(QualityReview)

    def analyze(self, description: str) -> AnalysisResult:
        result = self._analysis_chain.invoke({"description": description})
        return cast(AnalysisResult, result)

    def improve(
        self,
        original_description: str,
        current_description: str,
        analysis: str,
        review_feedback: str,
        attempt: int,
    ) -> ImprovementResult:
        result = self._improvement_chain.invoke(
            {
                "original_description": original_description,
                "current_description": current_description,
                "analysis": analysis,
                "review_feedback": review_feedback or "No previous review yet.",
                "attempt": attempt,
            }
        )
        return cast(ImprovementResult, result)

    def evaluate(
        self,
        original_description: str,
        improved_description: str,
    ) -> QualityReview:
        result = self._review_chain.invoke(
            {
                "original_description": original_description,
                "improved_description": improved_description,
            }
        )
        return cast(QualityReview, result)


def _format_list(items: list[str], empty_message: str) -> str:
    cleaned = [
        re.sub(r"^[-*•]+\s*", "", item.strip())
        for item in items
        if item and item.strip()
    ]
    if not cleaned:
        return f"- {empty_message}"
    return "\n".join(f"- {item}" for item in cleaned)


def format_analysis(result: AnalysisResult) -> str:
    """Turn structured analysis into readable UI text."""

    return (
        f"Summary\n{result.summary.strip()}\n\n"
        "Strengths\n"
        f"{_format_list(result.strengths, 'No clear strengths identified.')}\n\n"
        "Improvement opportunities\n"
        f"{_format_list(result.improvement_opportunities, 'No major gaps identified.')}"
    )


def format_review(result: QualityReview) -> str:
    """Turn a structured quality review into readable UI text."""

    sections = [result.rationale.strip()]
    if result.unsupported_claims:
        sections.append(
            "Unsupported claims\n"
            f"{_format_list(result.unsupported_claims, 'None identified.')}"
        )
    sections.append(
        "Remaining issues\n"
        f"{_format_list(result.remaining_issues, 'None — the copy is ready to use.')}"
    )
    return "\n\n".join(sections)


def find_unsupported_numeric_claims(
    original_description: str,
    improved_description: str,
) -> list[str]:
    """Detect numerical facts added by the model that are absent from the source."""

    number_pattern = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])")
    source_numbers = set(number_pattern.findall(original_description.lower()))
    candidate_numbers = set(number_pattern.findall(improved_description.lower()))
    added_numbers = sorted(candidate_numbers - source_numbers)
    return [
        f"The numeric claim '{number}' is not present in the original description."
        for number in added_numbers
    ]


def validate_description(description: object) -> str:
    """Validate and normalize user-supplied product copy."""

    if not isinstance(description, str) or not description.strip():
        raise WorkflowInputError(
            "Enter a product description before starting the review."
        )

    normalized = description.strip()
    if len(normalized) > MAX_DESCRIPTION_LENGTH:
        raise WorkflowInputError(
            f"Keep the product description under {MAX_DESCRIPTION_LENGTH:,} characters."
        )
    return normalized


def validate_max_retries(value: object) -> int:
    """Validate the maximum number of improvement attempts."""

    if isinstance(value, bool):
        raise WorkflowInputError(
            "Maximum attempts must be a whole number from 1 to 5."
        )

    try:
        numeric_value = float(cast(float | int | str, value))
    except (TypeError, ValueError) as exc:
        raise WorkflowInputError(
            "Maximum attempts must be a whole number from 1 to 5."
        ) from exc

    if not numeric_value.is_integer():
        raise WorkflowInputError(
            "Maximum attempts must be a whole number from 1 to 5."
        )

    max_retries = int(numeric_value)
    if not MIN_RETRIES <= max_retries <= MAX_RETRIES:
        raise WorkflowInputError(
            f"Maximum attempts must be between {MIN_RETRIES} and {MAX_RETRIES}."
        )
    return max_retries


def create_initial_state(description: object, max_retries: object) -> ProductState:
    """Create a complete and validated starting state for one graph run."""

    return {
        "product_description": validate_description(description),
        "analysis": "",
        "improved_description": "",
        "review_feedback": "",
        "quality_score": 0,
        "retry_count": 0,
        "max_retries": validate_max_retries(max_retries),
        "status": "pending",
        "trace": [],
    }


def route_review(
    state: ProductState,
) -> Literal["approved", "retry", "max_retries_reached"]:
    """Choose the next node based on quality and remaining attempts."""

    if state["quality_score"] >= APPROVAL_SCORE:
        return "approved"
    if state["retry_count"] < state["max_retries"]:
        return "retry"
    return "max_retries_reached"


def create_workflow(reviewer: ProductReviewer | None = None):
    """Build and compile the product-review graph."""

    service = reviewer or OllamaProductReviewer()

    def analyze(state: ProductState) -> dict[str, object]:
        result = service.analyze(state["product_description"])
        return {
            "analysis": format_analysis(result),
            "trace": ["Analyze: assessed the original description."],
        }

    def improve(state: ProductState) -> dict[str, object]:
        attempt = state["retry_count"] + 1
        current_description = (
            state["improved_description"] or state["product_description"]
        )
        result = service.improve(
            original_description=state["product_description"],
            current_description=current_description,
            analysis=state["analysis"],
            review_feedback=state["review_feedback"],
            attempt=attempt,
        )
        improved_description = result.description.strip()
        if not improved_description:
            raise RuntimeError("The model returned an empty improved description.")
        return {
            "improved_description": improved_description,
            "retry_count": attempt,
            "trace": [
                f"Improve: completed attempt {attempt} of {state['max_retries']}."
            ],
        }

    def review(state: ProductState) -> dict[str, object]:
        result = service.evaluate(
            original_description=state["product_description"],
            improved_description=state["improved_description"],
        )
        numeric_claims = find_unsupported_numeric_claims(
            state["product_description"], state["improved_description"]
        )
        unsupported_claims = list(dict.fromkeys(result.unsupported_claims + numeric_claims))
        # The LLM's score already accounts for qualitative unsupported claims.
        # Numeric additions receive a deterministic cap because they are an
        # especially risky and objectively detectable type of hallucination.
        score = min(result.score, 5) if numeric_claims else result.score
        effective_result = result.model_copy(
            update={"score": score, "unsupported_claims": unsupported_claims}
        )

        if score >= APPROVAL_SCORE:
            route = "approve"
        elif state["retry_count"] < state["max_retries"]:
            route = "retry"
        else:
            route = "manual review"
        return {
            "quality_score": score,
            "review_feedback": format_review(effective_result),
            "trace": [
                f"Review: scored {score}/10; conditional route = {route}."
            ],
        }

    def approved(_: ProductState) -> dict[str, object]:
        return {
            "status": "approved",
            "trace": ["Finish: content met the approval threshold."],
        }

    def max_retries_reached(_: ProductState) -> dict[str, object]:
        return {
            "status": "needs_manual_review",
            "trace": ["Finish: retry limit reached; manual review required."],
        }

    builder = StateGraph(ProductState)
    builder.add_node("analyze", analyze)
    builder.add_node("improve", improve)
    builder.add_node("review", review)
    builder.add_node("approved", approved)
    builder.add_node("max_retries_reached", max_retries_reached)

    builder.add_edge(START, "analyze")
    builder.add_edge("analyze", "improve")
    builder.add_edge("improve", "review")
    builder.add_conditional_edges(
        "review",
        route_review,
        {
            "approved": "approved",
            "retry": "improve",
            "max_retries_reached": "max_retries_reached",
        },
    )
    builder.add_edge("approved", END)
    builder.add_edge("max_retries_reached", END)
    return builder.compile()


# The compiled production graph is safe to reuse; each invocation supplies its
# own independent state.
graph = create_workflow()


def run_product_review(
    description: object,
    max_retries: object,
    workflow=None,
) -> ProductState:
    """Validate input and invoke the graph with a defensive recursion limit."""

    initial_state = create_initial_state(description, max_retries)
    selected_graph = workflow or graph
    # One analyze step, two steps per attempt, and one terminal step are expected.
    # The extra margin protects against accidental topology changes while the
    # state-based retry limit remains the primary loop guard.
    recursion_limit = (2 * initial_state["max_retries"]) + 5
    result = selected_graph.invoke(
        initial_state,
        config={"recursion_limit": recursion_limit},
    )
    return cast(ProductState, result)


if __name__ == "__main__":
    sample = (
        "The NovaBook Pro laptop includes 16GB RAM, a 512GB SSD, a 15.6-inch "
        "Full HD display, and a lightweight chassis for work and study."
    )
    final_state = run_product_review(sample, max_retries=3)
    print("\n========== FINAL RESULT ==========")
    print(f"Quality score: {final_state['quality_score']}/10")
    print(f"Attempts: {final_state['retry_count']}")
    print(f"Status: {final_state['status']}")
