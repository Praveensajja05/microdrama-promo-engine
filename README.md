# Automated Microdrama Promo Engine

An automated, series-agnostic Python pipeline and Streamlit dashboard that ingests full-season timestamped microdrama screenplays (`.zip` or `.txt`) and outputs ranked, Meta/Reels/TikTok-ready **20–60s video ad promos** backed by deterministic validation and an adversarial Zero-Context Cold-Scroll Critic.

---

## Architecture (5-Stage Hybrid Pipeline)

1. **Stage 1 — Batched Episode Scout (`Map`):** Parses screenplays into exact Beat IDs (`E01_B001`) with plain seconds (eliminating base-60 `MM:SS` LLM math errors) and scans 5 episodes per batch with a compact rolling story summary. Extracts continuous candidates and 10–22s `stitch_atoms`.
2. **Stage 2 & 2.5 — Deterministic Beat Validator + Smart Trimmer:** Verifies 100% of Beat IDs against the parsed script index. Oversized flagship scenes (60–220s) are routed to a surgical **Smart Trimmer** that outputs either a **Continuous** cut (25–55s) or a 2-part **Intra-Episode Cut** (keeping the 0–3s hook, skipping slow middle exposition, and jumping straight to the climax—enforced by ≥8s per segment and chronological non-overlap checks).
3. **Stage 3 — Zero-Context Cold-Scroll Critic:** Audits every candidate strictly from its isolated 20–60s transcript slice with **zero prior plot context**. Re-grounds `title`, `0–3s hook`, `cliffhanger`, and `why_it_works` from the slice, penalizes jarring `--- SCENE CUT ---` transitions, and applies a deterministic Python math gate (`min_score >= 3` and `total >= 11/15`).
4. **Stage 4A — Flash-Forward Cross-Episode Stitcher:** Pairs unused, non-overlapping `stitch_atoms` from distinct episodes into 24–55s flash-forward mashups with anti-recycling filters (<50% overlap with continuous winners) and a second Zero-Context Critic audit.
5. **Stage 4B — Global UA Ranker:** Ranks all verified promos (`Continuous`, `Intra-Episode Cut`, `Cross-Episode Stitched`) into launch tiers (`Wave 1 Launch Hero`, `Wave 2 Scale/Retarget`, `Wave 3 Test`), generates 0–3s overlay hooks, and exports verbatim **Script Proof (0–3s & last 3s)**.

---

## Verified Benchmarks (50 Episodes / 3,727 Beats)

| Iteration | Scouted | Trimmer Recovery | Ranked Promos (By Format) | Rejected Audit | Uncached Calls / Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **v1 (Baseline)** | 30 | 0/20 (67% lost) | 9 (8 Cont / 1 Cross) | 26 | 13 / 231K |
| **v2 (+ Plain Sec & Trimmer)** | 24 | 3/6 (50%) | 10 (9 Cont / 1 Cross) | 17 | 19 / 254K |
| **v3 (+ Meta DNA & Math Gate)** | 26 | 15/17 (88%) | 17 (14 Cont / 3 Cross) | 12 | 23 / 280K |
| **v4 (+ Intra-Episode Cuts)** | 26 | 8/10 (80%) | **20 (16 Cont / 2 Intra / 2 Cross)** | **9** | **23 / 297K (~110s)** |

*Primary LLM:* `nvidia/nemotron-3.5-lightning-30b-a3b` (`enable_thinking: False`, 45s fail-fast timeout) with automatic failover to `gemini-3.5-flash-lite` and optional `gpt-4o-mini`.

---

## 30-Second Quickstart

### 1. Setup Environment
```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure API Keys
Copy `.env.example` to `.env` and add your primary NVIDIA NIM key (plus optional Gemini/OpenAI backup keys):
```bash
cp .env.example .env
```

### 3. Run via CLI
```bash
python run_cli.py --input path/to/series.zip --provider nvidia
```
*(Add `--no-cache` to bypass prompt-fingerprinted disk caching and execute a fresh uncached run.)*

### 4. Launch Streamlit Dashboard
```bash
streamlit run app.py
```
* Upload any `.zip` archive or multi-file `.txt` episode scripts, or click **Load Latest Saved Run (`outputs/`)** in the sidebar to inspect pre-generated v4 results instantly.

---

## Repository Structure

```text
microdrama-promo-engine/
├── parser.py            # Regex screenplay parser & Beat ID normalizer
├── llm_client.py        # Multi-provider OpenAI-SDK wrapper with 45s timeout & failover
├── prompts.py           # Meta Ad Library calibrated prompts (Scout, Trimmer, Critic, Stitcher, Ranker)
├── pipeline.py          # 5-stage deterministic + agentic orchestration & exporters
├── run_cli.py           # Command-line runner
├── app.py               # Interactive Streamlit UI + CSV/Markdown/JSON downloads
├── outputs/             # Final v4 verified artifacts
│   ├── ranked_promos.csv
│   ├── promo_report.md
│   └── promos_raw.json
├── Solution_Doc.pdf     # 2-page technical architecture, Meta UA audit & failure log
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## UI Preview
*(Add your 2 collapsed-sidebar Streamlit screenshots in a `screenshots/` folder and link them below if pushing to GitHub.)*
* `screenshots/ranked_promos.png` — Ranked leaderboard with Critic `/15` scores, format tags, and verbatim script proof.
* `screenshots/quality_audit.png` — Rejected Audit log showing deterministic duration gates and Zero-Context Critic rejections.