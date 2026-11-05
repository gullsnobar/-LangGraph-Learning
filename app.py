"""Gradio interface for the LangGraph product-description reviewer."""

from __future__ import annotations

import html
import logging
from pathlib import Path

import gradio as gr

from retry_workflow import WorkflowInputError, run_product_review


LOGGER = logging.getLogger(__name__)
STYLES_PATH = Path(__file__).with_name("styles.css")
APP_CSS = STYLES_PATH.read_text(encoding="utf-8") if STYLES_PATH.exists() else ""


def _status_markup(kind: str, title: str, detail: str) -> str:
    """Render a small, accessible status panel with escaped content."""

    return (
        f'<div class="status-panel status-{html.escape(kind)}" role="status">'
        '<span class="status-dot" aria-hidden="true"></span>'
        '<div class="status-copy">'
        f"<strong>{html.escape(title)}</strong>"
        f"<span>{html.escape(detail)}</span>"
        "</div></div>"
    )


READY_STATUS = _status_markup(
    "ready",
    "Ready for review",
    "Enter a product description and choose the maximum improvement attempts.",
)


def _format_trace(trace: list[str]) -> str:
    if not trace:
        return "_The execution path will appear after a review._"
    lines = ["### Graph execution path"]
    lines.extend(f"{index}. {step}" for index, step in enumerate(trace, start=1))
    return "\n".join(lines)


def _friendly_runtime_error(error: Exception) -> str:
    """Translate common local-model failures into useful UI guidance."""

    messages: list[str] = []
    current: BaseException | None = error
    while current is not None and len(messages) < 4:
        messages.append(str(current))
        current = current.__cause__ or current.__context__
    combined = " ".join(messages).lower()

    if "connection refused" in combined or "connecterror" in combined:
        return (
            "Could not connect to Ollama. Start Ollama and confirm that "
            "http://127.0.0.1:11434 is available."
        )
    if "model" in combined and "not found" in combined:
        return (
            "The configured Ollama model is unavailable. Run "
            "`ollama pull llama3.2` or set OLLAMA_MODEL to an installed model."
        )
    if "validation error" in combined or "failed to parse" in combined:
        return (
            "The model returned an invalid structured response. Please try again; "
            "if the problem continues, use a model with reliable JSON support."
        )
    detail = str(error).strip()
    if detail:
        return f"The workflow stopped unexpectedly: {detail}"
    return "The workflow stopped unexpectedly. Check the terminal logs for details."


def review_product(description: str, max_retries: int):
    """Run the graph and convert its final state into Gradio output values."""

    try:
        result = run_product_review(description, max_retries)
    except WorkflowInputError as error:
        message = str(error)
        return (
            message,
            "",
            "",
            None,
            0,
            _status_markup("warning", "Input needed", message),
            "_The graph was not started because the input was invalid._",
        )
    except Exception as error:  # The UI must remain usable after model/runtime errors.
        LOGGER.exception("Product review failed")
        message = _friendly_runtime_error(error)
        return (
            "The review could not be completed.",
            "",
            message,
            None,
            0,
            _status_markup("error", "Review failed", message),
            "_The graph stopped because a runtime error occurred._",
        )

    status = result["status"]
    if status == "approved":
        status_markup = _status_markup(
            "success",
            "Approved",
            f"The description reached the {result['quality_score']}/10 quality score.",
        )
    else:
        status_markup = _status_markup(
            "warning",
            "Manual review recommended",
            "The retry limit was reached before the approval threshold.",
        )

    return (
        result["analysis"],
        result["improved_description"],
        result["review_feedback"],
        result["quality_score"],
        result["retry_count"],
        status_markup,
        _format_trace(result["trace"]),
    )


def reset_interface():
    """Restore every UI component to its initial value."""

    return (
        "",
        3,
        "",
        "",
        "",
        None,
        0,
        READY_STATUS,
        "_The execution path will appear after a review._",
    )


theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
    radius_size="lg",
)


with gr.Blocks(
    title="AI Content Quality Reviewer",
    fill_width=True,
) as demo:
    gr.HTML(
        """
        <header class="app-hero">
            <div class="eyebrow">LANGGRAPH WORKFLOW</div>
            <h1>AI Content Quality Reviewer</h1>
            <p>
                Analyze, refine, and score product descriptions through a
                transparent, retry-aware workflow.
            </p>
        </header>
        """
    )

    with gr.Row(equal_height=False, elem_classes=["workspace"]):
        with gr.Column(scale=5, min_width=320, elem_classes=["app-panel"]):
            gr.HTML(
                """
                <div class="section-heading">
                    <span>INPUT</span>
                    <h2>Product description</h2>
                    <p>Provide the factual source copy the workflow should review.</p>
                </div>
                """
            )
            description = gr.Textbox(
                label="Description",
                placeholder=(
                    "Example: The NovaBook Pro includes 16GB RAM, a 512GB SSD, "
                    "and a 15.6-inch Full HD display..."
                ),
                lines=11,
                max_lines=16,
                elem_id="description-input",
            )
            max_retries = gr.Slider(
                minimum=1,
                maximum=5,
                value=3,
                step=1,
                label="Maximum improvement attempts",
                info="The graph stops when approved or when this limit is reached.",
            )
            with gr.Row(elem_classes=["button-row"]):
                review_button = gr.Button(
                    "Review description",
                    variant="primary",
                    scale=3,
                    elem_id="review-button",
                )
                clear_button = gr.Button("Clear", variant="secondary", scale=1)
            gr.HTML(
                """
                <p class="privacy-note">
                    Runs locally through Ollama. The workflow does not invent
                    specifications that are absent from your source description.
                </p>
                """
            )

        with gr.Column(scale=7, min_width=360, elem_classes=["app-panel"]):
            gr.HTML(
                """
                <div class="section-heading">
                    <span>RESULT</span>
                    <h2>Quality review</h2>
                    <p>Inspect the final state produced by the graph.</p>
                </div>
                """
            )
            status = gr.HTML(value=READY_STATUS)
            with gr.Row(elem_classes=["metric-row"]):
                score = gr.Number(
                    label="Quality score (out of 10)",
                    interactive=False,
                    precision=0,
                    elem_classes=["metric-box"],
                )
                attempts = gr.Number(
                    value=0,
                    label="Improvement attempts",
                    interactive=False,
                    precision=0,
                    elem_classes=["metric-box"],
                )
            analysis = gr.Textbox(
                label="Content analysis",
                lines=7,
                max_lines=12,
                interactive=False,
            )
            improved = gr.Textbox(
                label="Improved description",
                lines=7,
                max_lines=12,
                interactive=False,
            )
            feedback = gr.Textbox(
                label="Quality review feedback",
                lines=5,
                max_lines=10,
                interactive=False,
            )
            with gr.Accordion("View workflow execution", open=False):
                trace = gr.Markdown(
                    value="_The execution path will appear after a review._"
                )

    review_button.click(
        fn=review_product,
        inputs=[description, max_retries],
        outputs=[analysis, improved, feedback, score, attempts, status, trace],
        api_name="review_product",
        concurrency_limit=1,
        show_progress="full",
    )
    clear_button.click(
        fn=reset_interface,
        inputs=None,
        outputs=[
            description,
            max_retries,
            analysis,
            improved,
            feedback,
            score,
            attempts,
            status,
            trace,
        ],
        queue=False,
    )


demo.queue(default_concurrency_limit=1, max_size=8)


if __name__ == "__main__":
    demo.launch(theme=theme, css=APP_CSS)
