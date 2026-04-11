import time

import requests
import streamlit as st

API_BASE = "http://localhost:8000"
POLL_INTERVAL = 3  # seconds

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔍",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

if "run_id" not in st.session_state:
    st.session_state.run_id = None
if "polling" not in st.session_state:
    st.session_state.polling = False
if "last_run_data" not in st.session_state:
    st.session_state.last_run_data = None


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def submit_run(query: str, competitors: list[str]) -> dict:
    resp = requests.post(
        f"{API_BASE}/research/run",
        json={"query": query, "competitors": competitors},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def get_run(run_id: int) -> dict:
    resp = requests.get(f"{API_BASE}/research/{run_id}", timeout=10)
    resp.raise_for_status()
    return resp.json()


def approve_run(run_id: int) -> dict:
    resp = requests.post(f"{API_BASE}/research/{run_id}/approve", timeout=10)
    resp.raise_for_status()
    return resp.json()


def reject_run(run_id: int, feedback: str | None) -> dict:
    resp = requests.post(
        f"{API_BASE}/research/{run_id}/reject",
        json={"feedback": feedback or None},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# UI components
# ---------------------------------------------------------------------------

def render_agent_badges(agents: list[dict]) -> None:
    if not agents:
        return
    cols = st.columns(len(agents))
    icons = {"web": "🌐", "reviews": "💬", "sales": "📊"}
    for col, agent in zip(cols, agents):
        status = agent["status"]
        colour = {"success": "green", "failed": "red", "timeout": "orange"}.get(status, "grey")
        icon = icons.get(agent["source"], "🤖")
        confidence = f'{agent["confidence"] * 100:.0f}%' if agent.get("confidence") else "—"
        col.markdown(
            f"""
            <div style="border:1px solid {colour}; border-radius:8px; padding:10px; text-align:center;">
                <div style="font-size:1.5rem">{icon}</div>
                <div><b>{agent['source'].capitalize()}</b></div>
                <div style="color:{colour}; text-transform:uppercase; font-size:0.75rem">{status}</div>
                <div style="font-size:0.85rem">Confidence: {confidence}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_draft(data: dict) -> None:
    draft = data.get("draft", {})
    if not draft:
        return

    confidence = draft.get("confidence", 0)
    confidence_pct = draft.get("confidence_pct", "—")
    rejection_count = draft.get("rejection_count", 0)

    st.divider()

    # Confidence meter
    col1, col2, col3 = st.columns([2, 1, 1])
    col1.subheader("Draft Recommendation")
    col2.metric("Confidence", confidence_pct)
    if rejection_count > 0:
        col3.metric("Revision", f"{rejection_count}/3")

    st.progress(float(confidence), text=f"Synthesis confidence: {confidence_pct}")

    # Executive summary
    st.markdown("### Executive Summary")
    st.info(draft.get("summary", ""))

    # Findings per source
    findings = draft.get("findings_by_source", {})
    if findings:
        st.markdown("### Findings by Source")
        tabs = st.tabs(["🌐 Web Intelligence", "💬 Customer Reviews", "📊 Sales Data"])
        with tabs[0]:
            st.write(findings.get("web") or "_No web findings available._")
        with tabs[1]:
            st.write(findings.get("reviews") or "_No review findings available._")
        with tabs[2]:
            st.write(findings.get("sales") or "_No sales findings available._")

        overlapping = findings.get("overlapping_signals", [])
        contradictory = findings.get("contradictory_signals", [])

        if overlapping:
            st.markdown("### ✅ Overlapping Signals")
            for s in overlapping:
                st.markdown(f"- {s}")

        if contradictory:
            st.markdown("### ⚠️ Contradictory Signals")
            for s in contradictory:
                st.markdown(f"- {s}")

        recommendation = findings.get("recommendation")
        if recommendation:
            st.markdown("### 💡 Recommendation")
            st.success(recommendation)


def render_approval_buttons(run_id: int) -> None:
    st.divider()
    st.markdown("### Your Decision")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Approve", type="primary", use_container_width=True):
            with st.spinner("Finalising..."):
                try:
                    approve_run(run_id)
                    st.session_state.polling = True
                    st.rerun()
                except requests.HTTPError as e:
                    st.error(f"Approval failed: {e.response.text}")

    with col2:
        with st.expander("❌ Reject with feedback"):
            feedback = st.text_area(
                "What should be improved?",
                placeholder="e.g. Focus more on pricing signals and less on features...",
                height=100,
            )
            if st.button("Submit Rejection", use_container_width=True):
                with st.spinner("Triggering re-synthesis..."):
                    try:
                        reject_run(run_id, feedback)
                        st.session_state.polling = True
                        st.rerun()
                    except requests.HTTPError as e:
                        st.error(f"Rejection failed: {e.response.text}")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_submit() -> None:
    st.title("🔍 AI Research Assistant")
    st.caption("Autonomous competitor intelligence, customer feedback analysis, and sales signals — synthesised into a product recommendation.")

    st.divider()

    with st.form("research_form"):
        query = st.text_area(
            "Research Question",
            placeholder="e.g. What are our competitors doing in AI-powered analytics, and what are customers asking us for?",
            height=100,
        )
        competitors_raw = st.text_input(
            "Competitors",
            placeholder="e.g. Notion, Linear, Asana",
        )
        submitted = st.form_submit_button("Run Research", type="primary", use_container_width=True)

    if submitted:
        competitors = [c.strip() for c in competitors_raw.split(",") if c.strip()]
        if not query or len(query) < 10:
            st.error("Please enter a research question (at least 10 characters).")
        elif not competitors:
            st.error("Please enter at least one competitor.")
        else:
            with st.spinner("Submitting research request..."):
                try:
                    result = submit_run(query, competitors)
                    st.session_state.run_id = result["run_id"]
                    st.session_state.polling = True
                    st.session_state.last_run_data = None
                    st.rerun()
                except requests.HTTPError as e:
                    st.error(f"Failed to submit: {e.response.text}")
                except requests.ConnectionError:
                    st.error("Cannot connect to the API. Is the server running on port 8000?")


def page_results(run_id: int) -> None:
    if st.button("← New Research"):
        st.session_state.run_id = None
        st.session_state.polling = False
        st.session_state.last_run_data = None
        st.rerun()

    st.title(f"Research Run #{run_id}")

    # Fetch current state
    try:
        data = get_run(run_id)
        st.session_state.last_run_data = data
    except requests.HTTPError as e:
        st.error(f"Error fetching run: {e.response.text}")
        return
    except requests.ConnectionError:
        st.error("Lost connection to API.")
        data = st.session_state.last_run_data or {}

    status = data.get("status", "unknown")

    # Status banner
    status_config = {
        "running":          ("⏳ Running — agents are gathering intelligence...", "info"),
        "pending_approval": ("✋ Awaiting your approval", "warning"),
        "completed":        ("✅ Research complete and approved", "success"),
        "failed":           ("❌ Research run failed", "error"),
    }
    msg, kind = status_config.get(status, (f"Status: {status}", "info"))
    getattr(st, kind)(msg)

    # Agent status badges (shown once available)
    agents = data.get("agents", [])
    if agents:
        st.markdown("#### Agent Status")
        render_agent_badges(agents)

    # Draft recommendation
    if status in ("pending_approval", "completed", "failed"):
        render_draft(data)

    if status == "completed":
        st.balloons()

    # Approve / Reject buttons
    if status == "pending_approval":
        render_approval_buttons(run_id)

    # Auto-poll while running or re-synthesising
    if status == "running" and st.session_state.polling:
        time.sleep(POLL_INTERVAL)
        st.rerun()


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if st.session_state.run_id is None:
    page_submit()
else:
    page_results(st.session_state.run_id)
