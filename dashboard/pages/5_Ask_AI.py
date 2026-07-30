from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT_STR = str(PROJECT_ROOT)

if PROJECT_ROOT_STR in sys.path:
    sys.path.remove(PROJECT_ROOT_STR)

sys.path.insert(0, PROJECT_ROOT_STR)

from dashboard.agent_engine import ask_ads_agent
from dashboard.app_config import APP_TITLE
from src.llm.manager import get_llm_runtime_info


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title=f"AI Copilot | {APP_TITLE}",
    page_icon="🤖",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_CHAT_MESSAGES = 20

QUICK_ANALYSES = [
    {
        "label": "Risk Analysis",
        "icon": "⚠️",
        "question": "Hangi kampanyalar riskli?",
        "description": "ROAS, bütçe ve model risklerini analiz et.",
    },
    {
        "label": "Budget Decisions",
        "icon": "💰",
        "question": "Hangi kampanyalara daha fazla bütçe vermeliyim?",
        "description": "Artırma, azaltma ve koruma kararlarını gör.",
    },
    {
        "label": "ROAS Analysis",
        "icon": "📈",
        "question": "En yüksek ve en düşük ROAS değerine sahip kampanyaları göster.",
        "description": "En iyi ve en zayıf ROAS sonuçlarını karşılaştır.",
    },
    {
        "label": "Revenue Opportunities",
        "icon": "🚀",
        "question": "Geliri en çok artıracak kampanyalar hangileri?",
        "description": "En yüksek tahmini gelir fırsatlarını sırala.",
    },
    {
        "label": "Profit Opportunities",
        "icon": "📊",
        "question": "Kârı en çok artıracak kampanyalar hangileri?",
        "description": "En yüksek tahmini kâr fırsatlarını göster.",
    },
    {
        "label": "Executive Summary",
        "icon": "🧭",
        "question": "Bana yönetici özeti ver.",
        "description": "En önemli kararları ve öncelikleri özetle.",
    },
]


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1320px;
            padding-top: 1.4rem;
            padding-bottom: 6rem;
        }

        .copilot-hero {
            position: relative;
            overflow: hidden;
            padding: 1.8rem 2rem;
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 22px;
            background:
                radial-gradient(
                    circle at top right,
                    rgba(59, 130, 246, 0.20),
                    transparent 34%
                ),
                radial-gradient(
                    circle at bottom left,
                    rgba(14, 165, 233, 0.10),
                    transparent 35%
                ),
                rgba(15, 23, 42, 0.92);
            margin-bottom: 1.2rem;
            box-shadow: 0 14px 40px rgba(2, 6, 23, 0.20);
        }

        .copilot-eyebrow {
            color: #7dd3fc;
            font-size: 0.76rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            margin-bottom: 0.45rem;
        }

        .copilot-title {
            font-size: clamp(2rem, 4vw, 3rem);
            line-height: 1.08;
            font-weight: 850;
            margin: 0;
        }

        .copilot-subtitle {
            color: #b8c3d1;
            line-height: 1.65;
            max-width: 880px;
            margin-top: 0.75rem;
            margin-bottom: 0;
        }

        .status-bar {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            padding: 0.8rem 1rem;
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 14px;
            background: rgba(15, 23, 42, 0.55);
            margin-bottom: 1.2rem;
        }

        .status-dot-online,
        .status-dot-offline {
            width: 10px;
            height: 10px;
            border-radius: 999px;
            flex: 0 0 auto;
        }

        .status-dot-online {
            background: #22c55e;
            box-shadow: 0 0 0 5px rgba(34, 197, 94, 0.12);
        }

        .status-dot-offline {
            background: #f59e0b;
            box-shadow: 0 0 0 5px rgba(245, 158, 11, 0.12);
        }

        .status-title {
            font-weight: 750;
        }

        .status-description {
            color: #94a3b8;
            font-size: 0.88rem;
        }

        .section-heading {
            font-size: 1.05rem;
            font-weight: 800;
            margin-top: 0.5rem;
            margin-bottom: 0.25rem;
        }

        .section-subtitle {
            color: #94a3b8;
            font-size: 0.9rem;
            margin-bottom: 0.8rem;
        }

        div[data-testid="stButton"] > button {
            border-radius: 14px;
            min-height: 46px;
            font-weight: 700;
            border: 1px solid rgba(148, 163, 184, 0.18);
            transition:
                transform 0.15s ease,
                border-color 0.15s ease,
                background-color 0.15s ease;
        }

        div[data-testid="stButton"] > button:hover {
            transform: translateY(-1px);
            border-color: rgba(125, 211, 252, 0.55);
        }

        div[data-testid="stChatMessage"] {
            border: 1px solid rgba(148, 163, 184, 0.13);
            border-radius: 18px;
            padding: 0.55rem 0.75rem;
            margin-bottom: 0.75rem;
            background: rgba(15, 23, 42, 0.35);
        }

        div[data-testid="stChatInput"] {
            border-top: 1px solid rgba(148, 163, 184, 0.10);
        }

        .read-only-note {
            color: #94a3b8;
            font-size: 0.82rem;
            line-height: 1.5;
        }

        .empty-state {
            padding: 1.25rem 1.4rem;
            border: 1px dashed rgba(148, 163, 184, 0.22);
            border-radius: 18px;
            background: rgba(15, 23, 42, 0.28);
            margin-top: 0.4rem;
            margin-bottom: 1rem;
        }

        .empty-state-title {
            font-size: 1.05rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
        }

        .empty-state-text {
            color: #94a3b8;
            line-height: 1.55;
            margin: 0;
        }

        @media (max-width: 768px) {
            .block-container {
                padding-top: 0.9rem;
            }

            .copilot-hero {
                padding: 1.35rem;
                border-radius: 18px;
            }

            .status-bar {
                align-items: flex-start;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------

def build_initial_message() -> dict[str, str]:
    """Return the default assistant message."""
    return {
        "role": "assistant",
        "content": (
            "Merhaba. Risk, bütçe, ROAS, gelir, kâr ve yönetici özeti "
            "analizlerini en güncel reklam çıktıları üzerinden yapabilirim. "
            "Bir hızlı analiz seçebilir veya kendi sorunuzu yazabilirsiniz."
        ),
    }


def ensure_session_state() -> None:
    """Initialize required session-state keys."""
    if "agent_messages" not in st.session_state:
        st.session_state["agent_messages"] = [
            build_initial_message()
        ]

    if "queued_agent_question" not in st.session_state:
        st.session_state["queued_agent_question"] = None


def trim_message_history() -> None:
    """Keep chat history within a safe maximum size."""
    messages = st.session_state.get("agent_messages", [])

    if len(messages) <= MAX_CHAT_MESSAGES:
        return

    st.session_state["agent_messages"] = messages[-MAX_CHAT_MESSAGES:]


def queue_question(question: str) -> None:
    """Queue a quick-analysis question."""
    st.session_state["queued_agent_question"] = question


def clear_chat() -> None:
    """Reset the chat to its initial state."""
    st.session_state["agent_messages"] = [
        build_initial_message()
    ]
    st.session_state["queued_agent_question"] = None


ensure_session_state()
runtime_info = get_llm_runtime_info()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="copilot-hero">
        <div class="copilot-eyebrow">
            AI Decision Intelligence Copilot
        </div>
        <h1 class="copilot-title">
            Advertising Analytics Copilot
        </h1>
        <p class="copilot-subtitle">
            Analyze campaign performance, budget optimization, ROAS,
            revenue, profitability and machine-learning recommendations
            through a single decision-support interface.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Status and controls
# ---------------------------------------------------------------------------

top_left, top_right = st.columns(
    [4.7, 1.3],
    vertical_alignment="center",
)

with top_left:
    if runtime_info.get("ready", False):
        provider = str(
            runtime_info.get(
                "provider",
                "Configured provider",
            )
        )
        model = str(
            runtime_info.get(
                "model",
                "Configured model",
            )
        )

        st.markdown(
            f"""
            <div class="status-bar">
                <div class="status-dot-online"></div>
                <div>
                    <div class="status-title">
                        Hybrid AI Mode
                    </div>
                    <div class="status-description">
                        Deterministic analytics + {provider} / {model}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="status-bar">
                <div class="status-dot-offline"></div>
                <div>
                    <div class="status-title">
                        Deterministic Decision Mode
                    </div>
                    <div class="status-description">
                        LLM is not configured. All supported analyses remain
                        available through the deterministic analytics engine.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with top_right:
    if st.button(
        "New Analysis",
        icon=":material/restart_alt:",
        width="stretch",
        type="secondary",
    ):
        clear_chat()
        st.rerun()


with st.expander(
    "System details",
    expanded=False,
):
    detail_col_1, detail_col_2, detail_col_3 = st.columns(3)

    with detail_col_1:
        st.metric(
            "Mode",
            (
                "Hybrid"
                if runtime_info.get("ready", False)
                else "Deterministic"
            ),
        )

    with detail_col_2:
        st.metric(
            "Provider",
            str(
                runtime_info.get(
                    "provider",
                    "Not configured",
                )
            ),
        )

    with detail_col_3:
        st.metric(
            "Model",
            str(
                runtime_info.get(
                    "model",
                    "Not configured",
                )
            ),
        )


# ---------------------------------------------------------------------------
# Quick analyses
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="section-heading">Quick analysis</div>
    <div class="section-subtitle">
        Start with a common decision question or write your own below.
    </div>
    """,
    unsafe_allow_html=True,
)

quick_columns = st.columns(3)

for index, item in enumerate(QUICK_ANALYSES):
    target_column = quick_columns[index % 3]

    with target_column:
        if st.button(
            f"{item['icon']}  {item['label']}",
            key=f"quick_analysis_{index}",
            help=item["description"],
            width="stretch",
        ):
            queue_question(item["question"])
            st.rerun()


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------

st.divider()

conversation_header_col, conversation_action_col = st.columns(
    [4.8, 1.2],
    vertical_alignment="center",
)

with conversation_header_col:
    st.markdown(
        """
        <div class="section-heading">Conversation</div>
        <div class="section-subtitle">
            Responses are generated from the latest available analytics outputs.
        </div>
        """,
        unsafe_allow_html=True,
    )

with conversation_action_col:
    if st.button(
        "Clear Chat",
        icon=":material/delete_sweep:",
        width="stretch",
    ):
        clear_chat()
        st.rerun()


messages = st.session_state["agent_messages"]

if len(messages) == 1:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-state-title">
                What would you like to analyze?
            </div>
            <p class="empty-state-text">
                Choose a quick analysis above or ask a business question such as
                “Hangi kampanyalar riskli?” or
                “Geliri en çok artıracak kampanyalar hangileri?”
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------------------------------------------------------------------
# Question handling
# ---------------------------------------------------------------------------

prefilled_question = st.session_state.pop(
    "queued_agent_question",
    None,
)

if not prefilled_question:
    prefilled_question = st.session_state.pop(
        "prefilled_question",
        None,
    )


chat_question = st.chat_input(
    "Ask a business question about campaigns, ROAS, budget, revenue or risk..."
)

user_question = chat_question or prefilled_question


if user_question:
    clean_question = str(user_question).strip()

    if clean_question:
        st.session_state["agent_messages"].append(
            {
                "role": "user",
                "content": clean_question,
            }
        )
        trim_message_history()

        with st.chat_message("user"):
            st.markdown(clean_question)

        with st.chat_message("assistant"):
            progress_placeholder = st.empty()

            progress_placeholder.info(
                "Loading campaign data and analytics outputs..."
            )

            with st.spinner(
                "Evaluating recommendations and preparing the response..."
            ):
                agent_response = ask_ads_agent(
                    question=clean_question,
                    history=st.session_state["agent_messages"],
                )

            progress_placeholder.empty()
            st.markdown(agent_response)

        st.session_state["agent_messages"].append(
            {
                "role": "assistant",
                "content": agent_response,
            }
        )
        trim_message_history()


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()

footer_left, footer_right = st.columns(
    [4.5, 1.5],
    vertical_alignment="center",
)

with footer_left:
    st.markdown(
        """
        <div class="read-only-note">
            Read-only decision support: recommendations are not automatically
            applied to Google Ads. Review business context before execution.
        </div>
        """,
        unsafe_allow_html=True,
    )

with footer_right:
    st.caption(
        f"Conversation limit: {MAX_CHAT_MESSAGES} messages"
    )
