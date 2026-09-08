"""
app.py
------
Interview Trainer Agent — Full Streamlit Application
AICTE 2026 Project · Problem Statement #22

Pages:
  1. Home
  2. Candidate Profile & Resume Upload
  3. Interview Configuration
  4. Interview Session  (adaptive mock interview)
  5. Final Report
"""

from __future__ import annotations

import os
import tempfile
import traceback
from typing import Any

import streamlit as st

# ── Page config — must be first Streamlit call ──────────────────────────────
st.set_page_config(
    page_title="Interview Trainer Agent",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* Structural styles only — no forced background/text colours so that
   Streamlit's own dark/light theme is fully respected. */

/* Question card — uses a subtle blue tint that works on both themes */
.ita-question {
    border-left: 5px solid #2563eb;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.3rem;
    margin-bottom: 1rem;
    font-size: 1.0rem;
    line-height: 1.65;
}

/* Section header underline */
.ita-header {
    font-size: 1.45rem;
    font-weight: 700;
    padding-bottom: 6px;
    border-bottom: 2px solid #2563eb;
    margin-bottom: 1rem;
}

/* Button font weight */
[data-testid="stButton"] button {
    font-weight: 600;
}

/* Hide Streamlit footer and menu */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _clr(score: int) -> str:
    """Return a hex colour for a 1-10 score."""
    if score >= 8:
        return "#16a34a"
    if score >= 6:
        return "#d97706"
    if score >= 4:
        return "#ea580c"
    return "#dc2626"


def _lbl(score: int) -> str:
    if score >= 9:
        return "Excellent"
    if score >= 7:
        return "Good"
    if score >= 5:
        return "Adequate"
    if score >= 3:
        return "Needs Work"
    if score > 0:
        return "Poor"
    return "Unanswered"


def _readiness_clr(level: str) -> str:
    return {
        "Excellent": "#16a34a",
        "Interview Ready": "#16a34a",
        "Almost Ready": "#d97706",
        "Needs Work": "#ea580c",
        "Not Ready": "#dc2626",
    }.get(level, "#475569")


# ── Session state init ────────────────────────────────────────────────────────

def _init_state() -> None:
    defaults: dict[str, Any] = {
        "page": "home",
        # Profile
        "_profile_data": None,        # lightweight dict before agent creation
        "_form_name": "",
        # Resume
        "resume_text": "",
        "resume_info": {},
        "resume_context": "",
        "resume_parsed": False,
        # Agent / session
        "profile": None,
        "agent": None,
        # Interview state
        "last_evaluation": None,
        "evaluations_shown": set(),   # set of q_idx already expanded
        # Model answer cache:  q_num (int) -> str
        "model_answers": {},
        "show_model_answer_for": None,  # q_num or None
        # Final report
        "final_report": None,
        # RAG
        "index_built": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ── Navigation ────────────────────────────────────────────────────────────────

def _go(page: str) -> None:
    st.session_state.page = page
    st.rerun()


# ── Cached resources (built once per Streamlit process) ──────────────────────

@st.cache_resource(show_spinner=False)
def _get_retriever():
    """Load or build FAISS index — runs only once per server process."""
    from rag.retriever import Retriever
    return Retriever(
        vector_store_path="data/vector_store",
        knowledge_dir="data/interview_knowledge",
    )


@st.cache_resource(show_spinner=False)
def _get_granite():
    """Return a shared GraniteClient — authenticates only once."""
    from llm.granite import GraniteClient
    return GraniteClient()


# ── Safe question cleaner ─────────────────────────────────────────────────────

def _clean_question(raw: str) -> str:
    """Sanitise raw Granite output so only the question text is shown.

    Guards against: empty string, dict/JSON bleed, leading JSON keys,
    prompt leakage, 'None', and missing '?'.
    """
    if not raw or not raw.strip():
        return "Could not generate a question. Please try again."

    text = raw.strip()

    # Strip common JSON/dict artefacts
    import re
    # Remove ```json ... ``` fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    # If it starts with '{' it's raw JSON — try to extract "question" key
    if text.startswith("{"):
        try:
            import json
            obj = json.loads(text)
            for key in ("question", "Question", "text", "content"):
                if key in obj and isinstance(obj[key], str) and obj[key].strip():
                    text = obj[key].strip()
                    break
            else:
                # Fallback: first string value
                for v in obj.values():
                    if isinstance(v, str) and v.strip():
                        text = v.strip()
                        break
        except Exception:
            pass

    # Strip leading "Q1:", "Question:", "QUESTION:", "1." etc.
    text = re.sub(r"^(?:QUESTION|Question|Q\d+)[:\.\s]+", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^\d+[\.\)]\s*", "", text).strip()

    # Guard against None-string or blank
    if not text or text.lower() in ("none", "null", "n/a", ""):
        return "Could not generate a question. Please try again."

    # Ensure it ends with a question mark
    if not text.endswith("?"):
        text = text.rstrip(".") + "?"

    return text


# ── Error display ─────────────────────────────────────────────────────────────

def _show_error(user_msg: str, exc: Exception | None = None) -> None:
    """Show a friendly error; put the traceback inside an expander."""
    st.error(user_msg)
    if exc is not None:
        with st.expander("Technical details (for debugging)", expanded=False):
            st.code(traceback.format_exc(), language="text")


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            '<div style="font-size:1.5rem;font-weight:800;color:#1e3a8a;">🎯 Interview Trainer</div>',
            unsafe_allow_html=True,
        )
        st.caption("AICTE 2026 · Problem Statement #22")
        st.markdown("---")

        nav_items = [
            ("🏠  Home",                "home"),
            ("👤  Candidate Profile",   "profile"),
            ("⚙️  Interview Config",     "config"),
            ("💬  Interview Session",   "interview"),
            ("📊  Final Report",        "report"),
        ]
        for label, key in nav_items:
            disabled = (
                (key == "config"    and not st.session_state.get("_profile_data")) or
                (key == "interview" and st.session_state.agent is None) or
                (key == "report"    and st.session_state.final_report is None)
            )
            if st.button(label, key=f"nav_{key}",
                         use_container_width=True, disabled=disabled):
                _go(key)

        st.markdown("---")

        # Show session progress if interview is live
        agent = st.session_state.agent
        if agent is not None:
            session = agent.get_session()
            answered = len(session.answers)
            total = len(session.questions)
            if total:
                st.progress(answered / total,
                            text=f"Progress: {answered}/{total} questions")

        st.markdown("---")
        st.caption("Powered by **IBM Granite 4 H Small**")
        st.caption("via IBM watsonx.ai · FAISS + sentence-transformers RAG")
        st.caption("AICTE 2026 · Problem Statement #22")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════

def _page_home() -> None:
    st.markdown(
        '<div class="ita-header">🎯 Interview Trainer Agent</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "**AICTE 2026 · Problem Statement #22** — "
        "AI-powered personalised mock interview preparation using IBM Granite."
    )
    st.markdown("")

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown("#### 🤖 IBM Granite AI")
        st.markdown(
            "Questions generated by **IBM Granite 4 H Small** "
            "via watsonx.ai, grounded in a curated knowledge base (RAG)."
        )
    with c2:
        st.markdown("#### 📄 Resume-Aware")
        st.markdown(
            "Upload your PDF resume. Granite extracts skills, projects, and "
            "experience to personalise every question."
        )
    with c3:
        st.markdown("#### 📊 Scored Feedback")
        st.markdown(
            "Each answer is scored 1–10 with strengths, weaknesses, and an "
            "improvement plan. Finish with a full readiness report."
        )

    st.markdown("---")
    st.markdown("### How it works")
    steps = [
        "Set up your **Candidate Profile** (name, role, experience)",
        "Upload an **optional PDF resume** for personalised questions",
        "Choose **Interview Type** (Technical / HR / Behavioral / Mixed) and **Difficulty**",
        "Start the **Adaptive Mock Interview** — Granite generates 8–10 questions",
        "Submit answers → get **real-time evaluation** from Granite",
        "Review your **Final Readiness Report** with improvement plan",
    ]
    for i, step in enumerate(steps, 1):
        st.markdown(f"**{i}.** {step}")

    st.markdown("---")

    with st.expander("🔌 Test IBM watsonx.ai Connection", expanded=False):
        st.caption(
            "Click below to verify that your API key is configured correctly "
            "and IBM Granite is reachable."
        )
        if st.button("Test Connection", key="test_conn"):
            with st.spinner("Connecting to IBM Granite..."):
                try:
                    client = _get_granite()
                    response = client.generate(
                        "Reply with exactly three words: 'Connection is OK'."
                    )
                    st.success(f"Connected successfully!  Granite replied: *{response}*")
                except Exception as exc:
                    _show_error(
                        "Connection failed. Check your .env credentials.",
                        exc,
                    )

    st.markdown("")
    if st.button("Get Started →", type="primary", key="btn_home_start"):
        _go("profile")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CANDIDATE PROFILE
# ══════════════════════════════════════════════════════════════════════════════

def _page_profile() -> None:
    st.markdown(
        '<div class="ita-header">👤 Candidate Profile &amp; Resume</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Fill in your details below. "
        "Resume upload is optional but improves question personalisation."
    )
    st.markdown("")

    col_left, col_right = st.columns([1.1, 1], gap="large")

    # ── Left: basic info ──────────────────────────────────────────────────────
    with col_left:
        st.markdown("#### Basic Information")

        name = st.text_input(
            "Full Name *",
            value=st.session_state.get("_form_name", ""),
            placeholder="e.g. Rahul Sharma",
            key="input_name",
        )

        role = st.selectbox(
            "Target Job Role *",
            options=[
                "Python Developer",
                "Data Scientist",
                "Machine Learning Engineer",
                "Software Developer",
                "Data Analyst",
                "AI Engineer",
            ],
            index=(
                ["Python Developer","Data Scientist","Machine Learning Engineer",
                 "Software Developer","Data Analyst","AI Engineer"].index(
                    st.session_state.get("_profile_data", {}).get("role", "Python Developer")
                )
                if st.session_state.get("_profile_data") else 0
            ),
            key="input_role",
        )

        experience = st.selectbox(
            "Experience Level *",
            options=["Fresher", "0-2 years", "2-5 years", "5+ years"],
            index=(
                ["Fresher","0-2 years","2-5 years","5+ years"].index(
                    st.session_state.get("_profile_data", {}).get("experience", "Fresher")
                )
                if st.session_state.get("_profile_data") else 0
            ),
            key="input_experience",
        )

        st.markdown("")
        if st.button("Save Profile & Continue →", type="primary",
                     key="btn_save_profile"):
            if not name.strip():
                st.error("Please enter your name.")
                return
            st.session_state._profile_data = {
                "name": name.strip(),
                "role": role,
                "experience": experience,
            }
            st.session_state._form_name = name.strip()
            st.success(f"Profile saved for **{name.strip()}**.")
            _go("config")

    # ── Right: resume ─────────────────────────────────────────────────────────
    with col_right:
        st.markdown("#### Resume Upload (Optional)")

        uploaded = st.file_uploader(
            "Upload PDF resume",
            type=["pdf"],
            help="Granite will extract your skills, projects, and experience.",
            key="resume_upload",
        )

        if uploaded is not None:
            file_kb = uploaded.size // 1024
            st.info(
                f"**{uploaded.name}** — {file_kb} KB  \n"
                "Click **Parse Resume** to extract information."
            )

            if st.button("Parse Resume", key="btn_parse"):
                with st.spinner("Extracting resume with IBM Granite..."):
                    tmp_path: str | None = None
                    try:
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".pdf"
                        ) as tmp:
                            tmp.write(uploaded.read())
                            tmp_path = tmp.name

                        from resume.parser import (
                            extract_resume_info,
                            format_resume_context,
                            parse_resume,
                        )
                        parsed = parse_resume(tmp_path)
                        info = extract_resume_info(parsed["text"])
                        ctx = format_resume_context(info)

                        st.session_state.resume_text = parsed["text"]
                        st.session_state.resume_info = info
                        st.session_state.resume_context = ctx
                        st.session_state.resume_parsed = True

                        st.success(
                            f"Resume parsed successfully — "
                            f"{parsed['pages']} page(s) processed."
                        )
                    except FileNotFoundError as exc:
                        st.error(str(exc))
                    except ValueError as exc:
                        st.warning(
                            f"Could not extract text: {exc}  \n"
                            "The interview will proceed without resume personalisation."
                        )
                    except Exception as exc:
                        _show_error(
                            "Resume parsing failed. "
                            "The interview will proceed without resume personalisation.",
                            exc,
                        )
                    finally:
                        if tmp_path and os.path.exists(tmp_path):
                            try:
                                os.unlink(tmp_path)
                            except OSError:
                                pass

        # Show extracted info card if available
        info = st.session_state.resume_info
        if info:
            st.markdown("##### Extracted Resume Information")
            rows: list[str] = []
            if info.get("name"):
                rows.append(f"**Name:** {info['name']}")
            if info.get("email"):
                rows.append(f"**Email:** {info['email']}")
            langs = [str(x) for x in info.get("programming_languages", []) if x]
            if langs:
                rows.append(f"**Languages:** {', '.join(langs[:8])}")
            fws = [str(x) for x in info.get("frameworks", []) if x]
            if fws:
                rows.append(f"**Frameworks:** {', '.join(fws[:8])}")
            skills = [str(x) for x in info.get("skills", []) if x]
            if skills:
                rows.append(f"**Skills:** {', '.join(skills[:8])}")
            projs = info.get("projects", [])
            if isinstance(projs, list) and projs:
                rows.append(f"**Projects:** {len(projs)} found")
            elif isinstance(projs, str) and projs:
                rows.append(f"**Projects:** {projs[:80]}")
            edu = info.get("education", [])
            if isinstance(edu, list) and edu:
                rows.append(f"**Education:** {edu[0][:80]}")
            elif isinstance(edu, str) and edu:
                rows.append(f"**Education:** {edu[:80]}")

            if rows:
                for row in rows:
                    st.markdown(row)
            else:
                st.caption(
                    "Resume was parsed but no structured fields were extracted. "
                    "Raw text will still be used to personalise questions."
                )

        elif st.session_state.resume_parsed:
            st.caption(
                "No structured information was extracted from the resume. "
                "Questions will still be personalised using the raw text."
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: INTERVIEW CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

def _page_config() -> None:
    profile_data = st.session_state.get("_profile_data")
    if not profile_data:
        st.warning("Please complete your candidate profile first.")
        if st.button("Go to Profile", key="cfg_go_profile"):
            _go("profile")
        return

    st.markdown(
        '<div class="ita-header">⚙️ Interview Configuration</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"Configuring for **{profile_data['name']}** · "
        f"{profile_data['role']} · {profile_data['experience']}"
    )
    st.markdown("")

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown("#### Interview Settings")

        interview_type = st.selectbox(
            "Interview Type",
            options=["Technical", "HR", "Behavioral", "Mixed"],
            help=(
                "**Technical** — programming, algorithms, system design  \n"
                "**HR** — background, goals, cultural fit  \n"
                "**Behavioral** — STAR-method past experiences  \n"
                "**Mixed** — combination of all three"
            ),
            key="cfg_type",
        )

        difficulty = st.selectbox(
            "Starting Difficulty",
            options=["Easy", "Medium", "Hard"],
            index=1,
            help=(
                "**Easy** — foundational concepts  \n"
                "**Medium** — applied knowledge and trade-offs  \n"
                "**Hard** — advanced depth and design decisions  \n"
                "Difficulty adapts automatically based on your performance."
            ),
            key="cfg_diff",
        )

        st.markdown("")
        st.markdown("#### Knowledge Base")
        st.caption(
            "The RAG index is built from `data/interview_knowledge/`. "
            "It loads automatically — click below to force a refresh."
        )

        rag_col1, rag_col2 = st.columns([2, 1])
        with rag_col2:
            if st.button("Rebuild Index", key="btn_rebuild_idx"):
                with st.spinner("Rebuilding RAG index..."):
                    try:
                        r = _get_retriever()
                        r.build_index()
                        st.session_state.index_built = True
                        st.success("Index rebuilt.")
                    except Exception as exc:
                        _show_error("Index build failed.", exc)
        with rag_col1:
            if st.session_state.index_built:
                st.success("Index ready.")
            else:
                st.info("Index will load automatically when the interview starts.")

    with col_r:
        st.markdown("#### Interview Summary")
        has_resume = bool(
            st.session_state.resume_info or st.session_state.resume_text
        )
        st.markdown(
            f"""
| Field | Value |
|---|---|
| Candidate | {profile_data['name']} |
| Role | {profile_data['role']} |
| Experience | {profile_data['experience']} |
| Interview Type | {interview_type} |
| Difficulty | {difficulty} (adaptive) |
| Resume | {"Uploaded ✓" if has_resume else "Not provided"} |
| Questions | 8–10 (adaptive) |
"""
        )
        st.caption(
            "Difficulty adjusts after each answer — strong answers unlock "
            "harder questions; weak answers trigger reinforcement."
        )

    st.markdown("---")

    if st.button("Start Interview →", type="primary", key="btn_start_interview"):
        with st.spinner(
            "Initialising session and generating your first question "
            "(this calls IBM Granite — may take a few seconds)..."
        ):
            try:
                from agents.interview_agent import CandidateProfile, InterviewAgent

                profile = CandidateProfile(
                    name=profile_data["name"],
                    role=profile_data["role"],
                    experience=profile_data["experience"],
                    interview_type=interview_type,
                    difficulty=difficulty,
                    resume_text=st.session_state.resume_text or "",
                    resume_info=st.session_state.resume_info or {},
                    resume_context=st.session_state.resume_context or "",
                )

                agent = InterviewAgent(profile=profile)
                agent.start_session()

                # Reset interview-specific state
                st.session_state.profile = profile
                st.session_state.agent = agent
                st.session_state.last_evaluation = None
                st.session_state.evaluations_shown = set()
                st.session_state.model_answers = {}
                st.session_state.show_model_answer_for = None
                st.session_state.final_report = None

                _go("interview")

            except EnvironmentError as exc:
                st.error(
                    "IBM watsonx.ai credentials are missing or invalid. "
                    f"Details: {exc}"
                )
            except Exception as exc:
                _show_error(
                    "Failed to start the interview session. "
                    "Check your .env credentials and internet connection.",
                    exc,
                )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: INTERVIEW SESSION
# ══════════════════════════════════════════════════════════════════════════════

def _page_interview() -> None:
    agent = st.session_state.agent
    if agent is None:
        st.warning("No active interview session. Please start a new one.")
        if st.button("Go to Configuration", key="int_go_config"):
            _go("config")
        return

    session = agent.get_session()
    profile = session.profile
    answered = len(session.answers)
    total_q = len(session.questions)

    # ── Page header ───────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="ita-header">💬 Mock Interview &mdash; {profile.name}</div>',
        unsafe_allow_html=True,
    )

    # Status bar
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Role", profile.role)
    h2.metric("Type", profile.interview_type)
    h3.metric("Difficulty", session._current_difficulty)
    h4.metric("Progress", f"{answered} / {total_q}")

    progress_pct = answered / max(total_q, 1)
    st.progress(progress_pct)
    st.markdown("")

    # ── Session complete ───────────────────────────────────────────────────────
    if session.is_complete:
        st.success(
            f"Interview complete! You answered {answered} questions. "
            "Generating your final report..."
        )
        if st.session_state.final_report is None:
            with st.spinner("Compiling report..."):
                try:
                    report = agent.generate_final_report()
                    st.session_state.final_report = report
                    _go("report")
                except Exception as exc:
                    _show_error("Report generation failed.", exc)
        else:
            if st.button("View Final Report →", type="primary",
                         key="int_view_report"):
                _go("report")
        return

    # ── Show previous evaluations (all answered so far) ───────────────────────
    for q_idx in range(answered):
        if q_idx not in st.session_state.evaluations_shown:
            # Mark new ones as auto-expanded once
            st.session_state.evaluations_shown.add(q_idx)
        _render_evaluation_card(q_idx, session, expanded=(q_idx == answered - 1))

    # ── Current question ───────────────────────────────────────────────────────
    q_num = answered + 1   # 1-based display number
    q_idx = session.current_question_index

    raw_question = session.questions[q_idx] if q_idx < len(session.questions) else ""
    question = _clean_question(raw_question)

    # Adaptive topic indicator
    current_topic = (
        session._covered_topics[-1] if session._covered_topics else "General"
    )
    q_type = (
        session._question_types[q_idx]
        if q_idx < len(session._question_types)
        else profile.interview_type
    )

    # Use _MAX_QUESTIONS from the agent module for a clean "N of ~8" display.
    # We import the constant directly to avoid coupling app.py to internal values.
    from agents.interview_agent import _MAX_QUESTIONS as _IA_MAX_Q
    st.markdown(
        f"**Question {q_num} of ~{_IA_MAX_Q}** &nbsp;"
        f"`{q_type}` &nbsp; `{session._current_difficulty}` &nbsp; Topic: *{current_topic}*"
    )

    st.markdown(
        f'<div class="ita-question">'
        f'<strong>Q{q_num}:</strong> {question}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── RAG transparency ──────────────────────────────────────────────────────
    with st.expander("📚 Retrieved Interview Knowledge (RAG context used)", expanded=False):
        try:
            retriever = _get_retriever()
            query = f"{profile.role} {q_type} {current_topic} interview question"
            rag_ctx = retriever.retrieve(query, top_k=3)
            if rag_ctx:
                # Show source names only — not full chunk content
                import re as _re
                sources = _re.findall(r"\[Source:\s*(.+?)\]", rag_ctx)
                unique_sources = list(dict.fromkeys(sources))
                st.caption(
                    f"**{len(unique_sources)} knowledge source(s) retrieved:**  "
                    + ",  ".join(f"`{s}`" for s in unique_sources)
                )
                st.caption(
                    "These knowledge-base passages were used to ground the "
                    "question in verified interview content."
                )
                # Show abbreviated passages
                for src, passage in zip(
                    sources,
                    [p.strip() for p in rag_ctx.split("---") if p.strip()],
                ):
                    preview = passage.replace(f"[Source: {src}]", "").strip()[:200]
                    st.markdown(f"**[{src}]** — {preview}...")
            else:
                st.caption("No specific knowledge retrieved for this query.")
        except Exception:
            st.caption("RAG retrieval unavailable.")

    # ── Answer input ──────────────────────────────────────────────────────────
    st.markdown("")
    answer = st.text_area(
        "Your Answer",
        height=170,
        placeholder=(
            "Write your answer here. "
            "Be specific — treat this as a real interview."
        ),
        key=f"ans_{q_num}",
    )

    # ── Action buttons ────────────────────────────────────────────────────────
    b1, b2, b3, b4 = st.columns([2, 1.5, 1.5, 2])

    with b1:
        submit_btn = st.button(
            "Submit Answer", type="primary", key=f"btn_submit_{q_num}"
        )
    with b2:
        skip_btn = st.button("Skip Question", key=f"btn_skip_{q_num}")
    with b3:
        show_ma = st.button(
            "Show Model Answer",
            key=f"btn_ma_{q_num}",
        )
    with b4:
        # Finish early only if min questions reached
        if answered >= 5:
            finish_btn = st.button(
                "Finish Interview", key=f"btn_finish_{q_num}"
            )
        else:
            finish_btn = False
            st.caption(f"Complete at least {5 - answered} more question(s) to finish.")

    # ── Model answer (cached per question) ────────────────────────────────────
    if show_ma:
        # Toggle: if already shown for this q_num, hide; else show
        if st.session_state.show_model_answer_for == q_num:
            st.session_state.show_model_answer_for = None
        else:
            st.session_state.show_model_answer_for = q_num

    if st.session_state.show_model_answer_for == q_num:
        # Fetch once and cache in session state
        if q_num not in st.session_state.model_answers:
            with st.spinner("Generating model answer with IBM Granite..."):
                try:
                    from agents.question_generator import generate_model_answer
                    retriever = _get_retriever()
                    rag_ctx = retriever.retrieve(
                        f"{profile.role} {q_type} {question[:120]}", top_k=3
                    )
                    ma = generate_model_answer(
                        question=question,
                        role=profile.role,
                        experience=profile.experience,
                        rag_context=rag_ctx,
                    )
                    st.session_state.model_answers[q_num] = ma if ma and ma.strip() else None
                except Exception as exc:
                    st.session_state.model_answers[q_num] = None
                    _show_error("Could not generate a model answer.", exc)

        ma_text = st.session_state.model_answers.get(q_num)
        if ma_text:
            st.info(f"**Model Answer:** {ma_text}")
        elif q_num in st.session_state.model_answers:
            st.warning("Model answer could not be generated. Please try again.")

    # ── Handle button actions ─────────────────────────────────────────────────
    if submit_btn:
        if not answer.strip():
            st.warning(
                "Please type your answer before submitting, "
                "or click **Skip Question** to move on."
            )
        else:
            _do_submit(agent, answer.strip())

    if skip_btn:
        _do_submit(agent, "[Skipped — no answer provided]")

    if finish_btn:
        # Force session complete and go to report
        with st.spinner("Finishing interview and compiling report..."):
            try:
                session.is_complete = True
                report = agent.generate_final_report()
                st.session_state.final_report = report
                _go("report")
            except Exception as exc:
                _show_error("Could not generate the final report.", exc)


def _do_submit(agent: Any, answer: str) -> None:
    """Submit answer to agent, store evaluation, rerun."""
    with st.spinner("Evaluating your answer with IBM Granite..."):
        try:
            evaluation = agent.submit_answer(answer)
            st.session_state.last_evaluation = evaluation
            st.session_state.show_model_answer_for = None
            st.rerun()
        except RuntimeError as exc:
            st.error(f"Session error: {exc}")
        except Exception as exc:
            _show_error(
                "Unable to evaluate your answer. Please try again.", exc
            )


def _render_evaluation_card(q_idx: int, session: Any, expanded: bool) -> None:
    """Render the evaluation card for question at index q_idx."""
    if q_idx >= len(session.evaluations):
        return

    ev = session.evaluations[q_idx]
    score = int(ev.get("score", 0))
    clr = _clr(score)
    lbl = _lbl(score)
    q_text = session.questions[q_idx] if q_idx < len(session.questions) else ""
    a_text = session.answers[q_idx] if q_idx < len(session.answers) else ""
    q_type = (
        session._question_types[q_idx]
        if q_idx < len(session._question_types)
        else "—"
    )

    with st.expander(
        f"Q{q_idx + 1} Evaluation — Score: {score}/10 ({lbl})  [{q_type}]",
        expanded=expanded,
    ):
        # Score badge + question
        st.markdown(
            f'<span style="display:inline-block;background:{clr};color:#fff;'
            f'font-weight:700;font-size:1.05rem;padding:3px 14px;'
            f'border-radius:20px;">{score}/10 — {lbl}</span>',
            unsafe_allow_html=True,
        )
        st.caption(f"Q: {q_text[:200]}")
        if a_text and a_text != "[Skipped — no answer provided]":
            st.caption(
                f"Your answer: {a_text[:150]}{'...' if len(a_text) > 150 else ''}"
            )
        else:
            st.caption("*(Skipped)*")

        # Strengths / Weaknesses
        s_col, w_col = st.columns(2)
        strengths = [str(s) for s in ev.get("strengths", []) if s]
        weaknesses = [str(w) for w in ev.get("weaknesses", []) if w]

        with s_col:
            if strengths:
                st.markdown("**Strengths**")
                for s in strengths:
                    st.markdown(f"- ✅ {s}")
            else:
                st.caption("No strengths identified.")

        with w_col:
            if weaknesses:
                st.markdown("**Areas to Improve**")
                for w in weaknesses:
                    st.markdown(f"- ⚠️ {w}")
            else:
                st.caption("No major weaknesses noted.")

        # Missing points
        missing = [str(m) for m in ev.get("missing_points", []) if m]
        if missing:
            st.markdown("**Missing Points**")
            for m in missing:
                st.markdown(f"- 📌 {m}")

        # Advice
        advice = str(ev.get("improvement_advice", "")).strip()
        if advice:
            st.info(f"**Improvement Advice:** {advice}")

        hint = str(ev.get("better_answer_hint", "")).strip()
        if hint:
            st.success(f"**Ideal Answer Outline:** {hint}")

        # STAR for behavioral
        star = ev.get("star_analysis")
        if star and isinstance(star, dict):
            st.markdown("**STAR Analysis**")
            sc1, sc2, sc3, sc4 = st.columns(4)
            for col, (key, title) in zip(
                [sc1, sc2, sc3, sc4],
                [("situation","Situation"),("task","Task"),
                 ("action","Action"),("result","Result")],
            ):
                with col:
                    val = str(star.get(key, "")).strip()
                    st.markdown(f"**{title}**")
                    st.caption(val if val else "Not addressed")

    st.markdown("")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FINAL REPORT
# ══════════════════════════════════════════════════════════════════════════════

def _page_report() -> None:
    report = st.session_state.final_report

    if report is None:
        # Try generating if session is complete but report wasn't stored
        agent = st.session_state.agent
        if agent is not None:
            session = agent.get_session()
            if session.is_complete:
                with st.spinner("Generating report..."):
                    try:
                        report = agent.generate_final_report()
                        st.session_state.final_report = report
                    except Exception as exc:
                        _show_error("Report generation failed.", exc)
                        return
        if report is None:
            st.warning("No report available. Complete the interview first.")
            if st.button("Go to Interview", key="rpt_go_int"):
                _go("interview")
            return

    name = report.get("candidate_name", "")
    role = report.get("role", "")
    overall = float(report.get("overall_score", 0))
    readiness = report.get("readiness_level", "Needs Work")
    r_clr = _readiness_clr(readiness)

    st.markdown(
        f'<div class="ita-header">📊 Interview Readiness Report — {name}</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"Role: {role}  ·  "
        f"Questions answered: {report.get('questions_answered', 0)} / "
        f"{report.get('total_questions', 0)}"
    )
    st.markdown("")

    # ── Readiness banner ──────────────────────────────────────────────────────
    st.markdown(
        f'<div style="background:{r_clr};color:#fff;border-radius:12px;'
        f'padding:1.5rem 2rem;margin-bottom:1.5rem;text-align:center;">'
        f'<div style="font-size:2.2rem;font-weight:800;">{readiness}</div>'
        f'<div style="font-size:1.05rem;opacity:0.9;margin-top:4px;">'
        f'Overall Score: {overall:.1f} / 10</div>'
        f'<div style="font-size:0.88rem;opacity:0.8;margin-top:6px;">'
        f'This is an AI-based practice evaluation, not a real hiring decision.'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Readiness scale
    st.markdown("**Score interpretation:**")
    scale_cols = st.columns(5)
    scale = [
        ("9–10", "Excellent",       "#16a34a"),
        ("7–8.9","Interview Ready", "#16a34a"),
        ("6–6.9","Almost Ready",    "#d97706"),
        ("4–5.9","Needs Work",      "#ea580c"),
        ("0–3.9","Not Ready",       "#dc2626"),
    ]
    for col, (rng, lvl, clr) in zip(scale_cols, scale):
        col.markdown(
            f'<div style="text-align:center;padding:6px;border-radius:6px;'
            f'border:1px solid {clr};">'
            f'<div style="font-weight:700;color:{clr};">{rng}</div>'
            f'<div style="font-size:0.8rem;">{lvl}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Score breakdown ───────────────────────────────────────────────────────
    st.markdown("### Score Breakdown")
    scores = {
        "Technical": float(report.get("technical_score", 0)),
        "Behavioral": float(report.get("behavioral_score", 0)),
        "HR": float(report.get("hr_score", 0)),
        "Communication": float(report.get("communication_score", 0)),
    }
    mcols = st.columns(4)
    for col, (label, val) in zip(mcols, scores.items()):
        clr = _clr(int(round(val)))
        col.markdown(
            f'<div style="text-align:center;padding:1rem;border-radius:10px;'
            f'border:1px solid {clr}33;">'
            f'<div style="font-size:2rem;font-weight:700;color:{clr};">'
            f'{val:.1f}</div>'
            f'<div style="font-size:0.88rem;">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ── Summary stats ─────────────────────────────────────────────────────────
    s1, s2, s3 = st.columns(3)
    s1.metric("Questions Attempted", report.get("questions_answered", 0))
    s2.metric("Questions Total", report.get("total_questions", 0))
    q_skipped = sum(
        1 for item in report.get("question_breakdown", [])
        if "[skipped" in str(item.get("answer_preview", "")).lower()
    )
    s3.metric("Questions Skipped", q_skipped)

    st.markdown("---")

    # ── Strong / Weak areas ───────────────────────────────────────────────────
    a_col, b_col = st.columns(2)
    with a_col:
        st.markdown("### ✅ Strong Areas")
        strong = report.get("strong_areas", [])
        if strong:
            for item in strong:
                st.markdown(f"- {item}")
        else:
            st.caption("Not enough data to identify strong areas.")

    with b_col:
        st.markdown("### ⚠️ Weak Areas")
        weak = report.get("weak_areas", [])
        if weak:
            for item in weak:
                st.markdown(f"- {item}")
        else:
            st.caption("Not enough data to identify weak areas.")

    st.markdown("---")

    # ── Topics to revise ─────────────────────────────────────────────────────
    topics = report.get("topics_to_revise", [])
    if topics:
        st.markdown("### 📚 Topics to Revise")
        for t in topics:
            st.markdown(f"- {t}")
        st.markdown("---")

    # ── Improvement plan ─────────────────────────────────────────────────────
    plan = str(report.get("improvement_plan", "")).strip()
    if plan:
        st.markdown("### 🗺️ Personalised Improvement Plan")
        st.info(plan)
        st.markdown("---")

    # ── Question breakdown ────────────────────────────────────────────────────
    breakdown = report.get("question_breakdown", [])
    if breakdown:
        st.markdown("### 📝 Question-by-Question Breakdown")
        for i, item in enumerate(breakdown, 1):
            q_score = int(item.get("score", 0))
            q_clr = _clr(q_score)
            q_lbl = _lbl(q_score)
            q_text = str(item.get("question", "")).strip()
            preview_title = (q_text[:75] + "...") if len(q_text) > 75 else q_text
            with st.expander(
                f"Q{i}: {preview_title}  —  {q_score}/10 ({q_lbl})",
                expanded=False,
            ):
                st.markdown(
                    f'<span style="display:inline-block;background:{q_clr};'
                    f'color:#fff;font-weight:700;padding:3px 12px;'
                    f'border-radius:20px;font-size:1rem;">'
                    f'{q_score}/10 — {q_lbl}</span>',
                    unsafe_allow_html=True,
                )
                ans_prev = str(item.get("answer_preview", "")).strip()
                if ans_prev and "[skipped" not in ans_prev.lower():
                    st.markdown(f"**Your answer:** {ans_prev}")
                elif "[skipped" in ans_prev.lower():
                    st.caption("*(Skipped)*")
                feedback = str(item.get("key_feedback", "")).strip()
                if feedback:
                    st.markdown(f"**Key feedback:** {feedback}")

    st.markdown("---")

    # ── Restart ───────────────────────────────────────────────────────────────
    if st.button("Start a New Interview", type="primary", key="btn_new_interview"):
        for k in [
            "agent", "profile", "last_evaluation", "final_report",
            "evaluations_shown", "model_answers", "show_model_answer_for",
            "resume_info", "resume_context", "resume_text", "resume_parsed",
            "_profile_data", "_form_name",
        ]:
            if k in st.session_state:
                del st.session_state[k]
        _init_state()
        _go("profile")


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    _render_sidebar()
    page = st.session_state.get("page", "home")
    dispatch = {
        "home":      _page_home,
        "profile":   _page_profile,
        "config":    _page_config,
        "interview": _page_interview,
        "report":    _page_report,
    }
    dispatch.get(page, _page_home)()


if __name__ == "__main__":
    main()
