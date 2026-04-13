import pandas as pd
import streamlit as st

from core.models import CausalLMWrapper
from core.analysis import speedup
from core.utils import pick_device
from core.prompting import format_question_for_family
from core.streaming_decode import (
    stream_baseline_greedy_decode,
    stream_speculative_greedy_decode,
)
from core.naive_multitoken import naive_multitoken_decode


MODEL_FAMILIES = {
    "gpt2": {
        "draft": "distilgpt2",
        "target": "gpt2",
        "label": "GPT-2 family",
        "recommended_k": 3,
        "recommended_tokens": 32,
        "recommended_mode": "hybrid",
    },
    "tinyllama": {
        "draft": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "target": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "label": "TinyLlama (LLaMA-family)",
        "recommended_k": 2,
        "recommended_tokens": 48,
        "recommended_mode": "hybrid",
    },
    "qwen": {
        "draft": "Qwen/Qwen2.5-1.5B-Instruct",
        "target": "Qwen/Qwen2.5-1.5B-Instruct",
        "label": "Qwen2.5-1.5B-Instruct",
        "recommended_k": 2,
        "recommended_tokens": 48,
        "recommended_mode": "hybrid",
    },
    "smollm2": {
    "draft": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "target": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "label": "SmolLM2 Draft/Target",
    "recommended_k": 3,
    "recommended_tokens": 32,
    "recommended_mode": "hybrid",
},
}


st.set_page_config(page_title="SpecuPipe", layout="wide")
st.title("SpecuPipe: True Streaming Speculative Decoding Dashboard")


with st.sidebar:
    st.header("Settings")
    family = st.selectbox("Model Family", ["gpt2", "tinyllama", "qwen","smollm2"], index=0)
    cfg = MODEL_FAMILIES[family]

    question = st.text_area(
        "Question",
        value="What is speculative decoding and why can it reduce inference latency?"
    )

    detail_level = st.selectbox("Answer Detail", ["short", "moderate", "detailed"], index=1)
    include_naive = st.checkbox("Include naive multi-token comparison", value=False)
    use_fast_defaults = st.checkbox("Use recommended fast settings", value=True)

    if use_fast_defaults:
        max_new_tokens = st.slider(
            "Max New Tokens",
            min_value=8,
            max_value=96,
            value=cfg["recommended_tokens"],
            step=8,
        )
        k = st.slider(
            "Initial Speculation Depth (k)",
            min_value=2,
            max_value=8,
            value=cfg["recommended_k"],
            step=1,
        )
        mode = st.selectbox(
            "Mode",
            ["fixed", "adaptive", "hybrid"],
            index=["fixed", "adaptive", "hybrid"].index(cfg["recommended_mode"]),
        )
    else:
        max_new_tokens = st.slider("Max New Tokens", min_value=8, max_value=96, value=32, step=8)
        k = st.slider("Initial Speculation Depth (k)", min_value=2, max_value=8, value=3, step=1)
        mode = st.selectbox("Mode", ["fixed", "adaptive", "hybrid"])

    device = st.selectbox("Device", ["mps", "cpu"])
    run_button = st.button("Run Comparison")


@st.cache_resource
def load_model(model_name: str, device: str):
    return CausalLMWrapper(model_name, device=device)


def strip_prompt_from_output(prompt_text: str, generated_text: str) -> str:
    if generated_text.startswith(prompt_text):
        return generated_text[len(prompt_text):].strip()
    return generated_text.strip()


if run_button:
    actual_device = pick_device(device)
    family_cfg = MODEL_FAMILIES[family]
    formatted_prompt = format_question_for_family(question, family, detail_level)

    with st.spinner("Loading models..."):
        draft = load_model(family_cfg["draft"], actual_device)
        target = load_model(family_cfg["target"], actual_device)

    eos_id = target.tokenizer.eos_token_id

    st.subheader("Live Decode View")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Baseline Answer")
        baseline_placeholder = st.empty()
        baseline_status = st.empty()

    with c2:
        st.markdown("### Speculative Answer")
        spec_placeholder = st.empty()
        spec_status = st.empty()

    baseline_metrics = None
    spec_metrics = None
    cache = None
    baseline_text = ""
    spec_text = ""

    for event in stream_baseline_greedy_decode(
        model=target,
        prompt=formatted_prompt,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_id,
    ):
        if event["type"] == "token":
            baseline_text = event["text"]
            baseline_answer = strip_prompt_from_output(formatted_prompt, baseline_text)
            baseline_placeholder.markdown(baseline_answer + " ▌")
            baseline_status.info("Baseline is generating token by token...")
        else:
            baseline_text = event["text"]
            baseline_metrics = event["metrics"]
            baseline_answer = strip_prompt_from_output(formatted_prompt, baseline_text)
            baseline_placeholder.markdown(baseline_answer)
            baseline_status.success("Baseline finished.")

    for event in stream_speculative_greedy_decode(
        draft_model=draft,
        target_model=target,
        prompt=formatted_prompt,
        max_new_tokens=max_new_tokens,
        speculation_depth=k,
        eos_token_id=eos_id,
        adaptive=(mode in ["adaptive", "hybrid"]),
        hybrid_gated=(mode == "hybrid"),
        min_depth=2,
        max_depth=8,
    ):
        if event["type"] == "token":
            spec_text = event["text"]
            spec_answer = strip_prompt_from_output(formatted_prompt, spec_text)
            spec_placeholder.markdown(spec_answer + " ▌")

            if "accepted" in event:
                spec_status.info(
                    f"Round {event['round_id']}: accepted {event['accepted']} / drafted {event['drafted']}"
                )
            else:
                spec_status.info(f"Round {event['round_id']}: baseline fallback step")
        else:
            spec_text = event["text"]
            spec_metrics = event["metrics"]
            cache = event["cache"]
            spec_answer = strip_prompt_from_output(formatted_prompt, spec_text)
            spec_placeholder.markdown(spec_answer)
            spec_status.success("Speculative decoding finished.")

    sp = speedup(baseline_metrics.total_time_sec, spec_metrics.total_time_sec)
    output_match = baseline_text == spec_text

    st.subheader("Run Summary")
    a, b, c, d = st.columns(4)
    a.metric("Family", family_cfg["label"])
    b.metric("Speedup vs Baseline", f"{sp:.3f}x")
    c.metric("Output Match", "Yes" if output_match else "No")
    d.metric("Acceptance Rate", f"{spec_metrics.acceptance_rate:.3f}")

    e, f, g = st.columns(3)
    e.metric("Rollbacks", f"{spec_metrics.rollback_count}")
    f.metric("Wasted Draft Tokens", f"{spec_metrics.wasted_draft_tokens}")
    g.metric("KV Overhead Units", f"{spec_metrics.peak_estimated_kv_overhead_units:.0f}")

    st.subheader("Metrics Comparison")
    rows = [
        {
            "Mode": "Baseline",
            "Time (s)": baseline_metrics.total_time_sec,
            "Generated Tokens": baseline_metrics.generated_tokens,
            "Tokens/sec": baseline_metrics.tokens_per_second,
            "Matches Baseline": True,
        },
        {
            "Mode": f"Speculative ({mode})",
            "Time (s)": spec_metrics.total_time_sec,
            "Generated Tokens": spec_metrics.generated_tokens,
            "Tokens/sec": spec_metrics.tokens_per_second,
            "Matches Baseline": output_match,
        }
    ]

    naive_text = None
    naive_metrics = None
    naive_match = None
    naive_speedup = None

    if include_naive:
        with st.spinner("Running naive multi-token comparison..."):
            naive_text, _, naive_metrics = naive_multitoken_decode(
                model=draft,
                prompt=formatted_prompt,
                max_new_tokens=max_new_tokens,
                block_size=k,
                eos_token_id=eos_id,
            )
        naive_match = naive_text == baseline_text
        naive_speedup = speedup(baseline_metrics.total_time_sec, naive_metrics.total_time_sec)
        rows.append(
            {
                "Mode": "Naive Multi-Token",
                "Time (s)": naive_metrics.total_time_sec,
                "Generated Tokens": naive_metrics.generated_tokens,
                "Tokens/sec": naive_metrics.tokens_per_second,
                "Matches Baseline": naive_match,
            }
        )

    metrics_df = pd.DataFrame(rows)
    st.dataframe(metrics_df, use_container_width=True)

    if include_naive:
        with st.expander("Naive Multi-Token Comparison", expanded=False):
            n1, n2 = st.columns(2)
            with n1:
                st.markdown("### Naive Multi-Token Answer")
                st.write(strip_prompt_from_output(formatted_prompt, naive_text))
            with n2:
                st.markdown("### Why it matters")
                st.write(
                    "Naive multi-token decoding commits drafted tokens directly without target-model "
                    "verification. It can look attractive for speed, but it may drift away from the "
                    "baseline target-model output."
                )

            x, y, z = st.columns(3)
            x.metric("Naive Speedup vs Baseline", f"{naive_speedup:.3f}x")
            y.metric("Naive Matches Baseline", "Yes" if naive_match else "No")
            z.metric("Naive Tokens/sec", f"{naive_metrics.tokens_per_second:.3f}")

    st.subheader("Per-Round Timeline")
    rounds_df = pd.DataFrame([r.__dict__ for r in spec_metrics.step_records])
    st.dataframe(rounds_df, use_container_width=True)

    st.subheader("KV-Cache State History")
    cache_df = pd.DataFrame([e.__dict__ for e in cache.history])
    st.dataframe(cache_df, use_container_width=True)
