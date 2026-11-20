import csv
import io
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from llm_client import PROVIDER_PRESETS, SwappableLLMClient
from parser import load_episodes_from_bytes
from pipeline import PromoPipeline, export_outputs

load_dotenv()

st.set_page_config(page_title="Microdrama Promo Engine", layout="wide")
st.title("Microdrama Promo Engine")
st.caption(
    "Upload timestamped screenplay .txt files or a .zip archive -> get 20-60s ranked ad promos + automated quality audit."
)


def build_csv_bytes(result: dict) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "Rank", "ID", "Tier", "Title", "Type", "Total_Sec",
        "Segments (Episode & Timestamps)", "Burned-In 0-3s Overlay",
        "Hook (0-3s)", "Cliffhanger", "Why It Works", "Ranking Reason",
        "Critic Scores (H/C/Cl)", "Critic Total (/15)",
    ])
    for p in result.get("ranked_promos", []):
        segs_str = " + ".join(
            f"Ep {s['episode']} [{s['start_ts']} - {s['end_ts']}]" for s in p.get("segments", [])
        )
        cr = p.get("critic", {})
        c_score = f"{cr.get('hook_3s', '-')}/{cr.get('standalone_clarity', '-')}/{cr.get('cliffhanger_pull', '-')}"
        writer.writerow([
            p.get("rank", ""), p.get("candidate_id", ""), p.get("posting_tier", ""),
            p.get("title", ""), p.get("promo_format") or ("Cross-Episode Stitched" if p.get("is_stitched") else "Continuous"),
            p.get("total_duration_sec", ""), segs_str, p.get("overlay_hook", ""),
            p.get("hook_0_3s", ""), p.get("cliffhanger", ""), p.get("why_it_works", ""),
            p.get("ranking_reason", ""), c_score, p.get("critic_total", "-"),
        ])
    return buf.getvalue().encode("utf-8")


with st.sidebar:
    st.header("1. Primary Model (Nemotron)")
    provider = st.selectbox("Primary Provider", list(PROVIDER_PRESETS.keys()), index=0)
    preset = PROVIDER_PRESETS[provider]
    api_key = st.text_input(
        f"{provider.upper()} API Key",
        value=os.getenv(preset["env_key"], ""),
        type="password",
        key=f"api_key_{provider}",
    )
    scout_model = st.text_input(
        "Scout Model (Map)",
        value=preset["default_model"],
        key=f"scout_model_{provider}",
    )
    strong_model = st.text_input(
        "Critic & Ranker Model",
        value=preset["strong_model"],
        key=f"strong_model_{provider}",
    )

    st.divider()
    st.header("2. Automatic Backups (Optional)")
    gemini_backup_key = st.text_input(
        "Backup 1: Gemini API Key",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
    )
    openai_backup_key = st.text_input(
        "Backup 2: OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
    )

    st.divider()
    batch_size = st.slider("Episodes per Scout batch", 2, 10, 5)
    use_cache = st.checkbox("Use disk cache (resume on re-run)", value=True)

    latest_json = Path("outputs/promos_raw.json")
    if latest_json.exists():
        st.divider()
        if st.button("Load Latest Saved Run (outputs/)", use_container_width=True):
            st.session_state["result"] = json.loads(latest_json.read_text(encoding="utf-8"))
            st.success("Loaded outputs/promos_raw.json")

uploaded = st.file_uploader(
    "Upload episode scripts (.zip archive or multiple .txt files)",
    type=["zip", "txt"],
    accept_multiple_files=True,
)

if uploaded:
    episodes = load_episodes_from_bytes([(f.name, f.read()) for f in uploaded])
    st.success(
        f"Parsed **{len(episodes)} episodes** ({sum(len(e.beats) for e in episodes):,} timestamped beats)."
    )

    with st.expander("Parser Diagnostics (Verify 0 Dropped Episodes)"):
        st.dataframe(
            pd.DataFrame([
                {
                    "Episode": e.episode_num,
                    "Beats": len(e.beats),
                    "Duration (s)": e.duration_sec,
                    "Mode": e.parser_mode,
                }
                for e in episodes
            ]),
            use_container_width=True,
            hide_index=True,
        )

    if not api_key:
        st.info(f"Add your {provider.upper()} API Key in the sidebar (or .env) to enable execution.")

    if st.button("Find & Rank Promos", type="primary", disabled=not api_key):
        prog = st.progress(0.0)
        status = st.empty()
        try:
            backups = []
            if gemini_backup_key and provider != "gemini":
                backups.append({"provider": "gemini", "api_key": gemini_backup_key})
            if openai_backup_key and provider != "openai":
                backups.append({"provider": "openai", "api_key": openai_backup_key})

            llm = SwappableLLMClient(
                provider=provider,
                api_key=api_key,
                model=scout_model,
                strong_model=strong_model,
                backups=backups,
            )
            pipe = PromoPipeline(
                episodes=episodes,
                llm=llm,
                strong_model=strong_model,
                batch_size=batch_size,
            )
            res = pipe.run(
                progress_cb=lambda msg, frac: (status.write(msg), prog.progress(frac)),
                use_cache=use_cache,
            )
            res["meta"]["failover_events"] = llm.failover_events
            export_outputs(res, out_dir="outputs")
            st.session_state["result"] = res
        except Exception as e:
            st.error(f"Pipeline error: {e}")

if "result" in st.session_state:
    res = st.session_state["result"]
    m = res["meta"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Ranked Promos", m.get("final_ranked_count", 0))
    c2.metric("Rejected (Audit)", m.get("rejected_count", 0))
    c3.metric(
        "Trimmer Recovered",
        f"{m.get('recovered_by_trimmer', 0)}/{m.get('oversized_sent_to_trimmer', 0)}",
    )
    c4.metric("API Calls", m.get("llm_calls", 0))
    c5.metric("Total Tokens", f"{m.get('prompt_tokens', 0) + m.get('completion_tokens', 0):,}")

    if m.get("failover_events"):
        st.warning("Automatic failover triggered during run:\n- " + "\n- ".join(m["failover_events"]))

    dl1, dl2, dl3 = st.columns(3)
    dl1.download_button(
        "Download ranked_promos.csv",
        data=build_csv_bytes(res),
        file_name="ranked_promos.csv",
        mime="text/csv",
        use_container_width=True,
    )
    md_path = Path("outputs/promo_report.md")
    if md_path.exists():
        dl2.download_button(
            "Download promo_report.md",
            data=md_path.read_text(encoding="utf-8"),
            file_name="promo_report.md",
            mime="text/markdown",
            use_container_width=True,
        )
    dl3.download_button(
        "Download promos_raw.json",
        data=json.dumps(res, indent=2),
        file_name="promos_raw.json",
        mime="application/json",
        use_container_width=True,
    )

    tab1, tab2, tab3 = st.tabs(["Ranked Promos", "Quality Audit (Rejected)", "Raw JSON"])
    with tab1:
        for p in res["ranked_promos"]:
            segs = "  +  ".join(
                f"Ep {s['episode']} (`{s['start_ts']} - {s['end_ts']}`)" for s in p["segments"]
            )
            trim_note = f" | trimmed from {p['trimmed_from_sec']}s" if p.get("trimmed_from_sec") else ""
            fmt_tag = p.get("promo_format") or ("Cross-Episode Stitched" if p.get("is_stitched") else "Continuous")
            score_tag = f" | Critic {p['critic_total']}/15" if p.get("critic_total") else ""
            header = (
                f"#{p['rank']} [{p['posting_tier']}] {p['title']} — "
                f"{p['total_duration_sec']}s ({fmt_tag}{trim_note}{score_tag})"
            )
            with st.expander(header, expanded=(p["rank"] <= 5)):
                st.markdown(f"**Segments:** {segs}")
                if p.get("overlay_hook"):
                    st.markdown(f"**Suggested 0-3s Text Overlay:** `\"{p['overlay_hook']}\"`")
                st.markdown(f"**Verified 0-3s Hook:** {p['hook_0_3s']}")
                st.markdown(f"**Verified Cliffhanger:** {p['cliffhanger']}")
                st.markdown(f"**Why it works:** {p['why_it_works']}")
                if p.get("ranking_reason"):
                    st.markdown(f"**Ranking reason:** {p['ranking_reason']}")
                if p.get("critic"):
                    cr = p["critic"]
                    st.caption(
                        f"Cold-Scroll Critic -> Hook: {cr.get('hook_3s')}/5 | "
                        f"Clarity: {cr.get('standalone_clarity')}/5 | "
                        f"Cliffhanger: {cr.get('cliffhanger_pull')}/5 — {cr.get('critique', '')}"
                    )
                st.code(
                    "\n\n--- CUT ---\n\n".join(s["transcript"] for s in p["segments"]),
                    language="text",
                )

    with tab2:
        audit_rows = []
        for r in res.get("rejected_audit", []):
            sc = r.get("scores") or {}
            score_str = (
                f"{sc.get('hook_3s', '-')}/{sc.get('standalone_clarity', '-')}/{sc.get('cliffhanger_pull', '-')}"
                if sc
                else "-"
            )
            audit_rows.append({
                "ID": r.get("candidate_id", ""),
                "Title": r.get("title", ""),
                "Stage": r.get("stage", ""),
                "Scores (H/C/Cl)": score_str,
                "Segments": ", ".join(r.get("segments_preview", [])) if r.get("segments_preview") else "-",
                "Reason": r.get("reason", ""),
            })
        st.dataframe(pd.DataFrame(audit_rows), use_container_width=True, hide_index=True)

    with tab3:
        st.json(res, expanded=False)