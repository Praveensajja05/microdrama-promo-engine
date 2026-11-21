# Automated Microdrama Promo Engine

A Python pipeline and Streamlit dashboard that reads timestamped microdrama screenplays (`.zip` or `.txt`) and returns ranked **20–60s promo clips** (episode + start/end timestamps, hook, cliffhanger, why it works). Code handles parsing, timestamps and every hard rule; the LLM handles judgement; a human editor makes the final pick.

> **Two runs are included, both unedited.** `outputs/` is the v4 run on the full 50-episode pack. `outputs_v5/` is the v5 run after the fixes listed in [v5 Fixes](#v5-fixes-found-by-re-checking-the-v4-output) — including the problems v5 still has.

---

## Architecture (5-Stage Hybrid Pipeline)

1. **Stage 1 — Batched Episode Scout (`Map`):** Parses screenplays into exact Beat IDs (`E01_B001`) with plain seconds (eliminating base-60 `MM:SS` LLM math errors) and scans 5 episodes per batch with a compact rolling story summary. Extracts continuous candidates and 10–22s `stitch_atoms`.
2. **Stage 2 & 2.5 — Deterministic Beat Validator + Smart Trimmer:** Verifies 100% of Beat IDs against the parsed script index. Oversized flagship scenes (60–220s) are routed to a surgical **Smart Trimmer** that outputs either a **Continuous** cut (25–55s) or a 2-part **Intra-Episode Cut** (keeping the 0–3s hook, skipping slow middle exposition, and jumping straight to the climax—enforced by ≥8s per segment and chronological non-overlap checks).
3. **Stage 3 — Zero-Context Cold-Scroll Critic:** Audits every candidate strictly from its isolated 20–60s transcript slice with **zero prior plot context**. Re-grounds `title`, `0–3s hook`, `cliffhanger`, and `why_it_works` from the slice, penalizes jarring `--- SCENE CUT ---` transitions, and applies a deterministic Python math gate (`min_score >= 3` and `total >= 11/15`).
4. **Stage 4A — Flash-Forward Cross-Episode Stitcher:** Pairs unused, non-overlapping `stitch_atoms` from distinct episodes into 24–55s flash-forward mashups with anti-recycling filters (<50% overlap with continuous winners) and a second Zero-Context Critic audit.
5. **Stage 4B — Global Ranker:** Ranks all verified promos (`Continuous`, `Intra-Episode Cut`, `Cross-Episode Stitched`) into launch tiers (`Wave 1 Launch Hero`, `Wave 2 Scale/Retarget`, `Wave 3 Test`), generates 0–3s overlay hooks, and exports verbatim **Script Proof (0–3s & last 3s)**.
6. **Stage 4C — Per-Promo Overlay Writer + Grounding Audit (new in v5):** Each 0–3s overlay is written in its own LLM call from **only that promo's clip** (v4 wrote all 20 in one ranker call and cross-wired some). The model must first state *who does what to whom* before writing the overlay. Then `grounding.py` (pure code, no LLM) flags any promo whose overlay/hook/cliffhanger names a place or person not in its own clip, or whose hook/cliffhanger text doesn't match the clip's actual first/last 3 seconds. Flags go to the human reviewer; nothing is auto-deleted.

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

**Optional — independent critic:** score clips with a *different* model so the scout isn't grading its own picks:
```bash
python run_cli.py --input path/to/series.zip --provider nvidia --critic-provider gemini
```

### 4. Launch Streamlit Dashboard
```bash
streamlit run app.py
```
* Upload any `.zip` archive or multi-file `.txt` episode scripts, or click **Load Latest Saved Run (`outputs/`)** in the sidebar to inspect pre-generated v4 results instantly.

---

## Checking Quality Without Spending Tokens

```bash
# Audit any existing run against the scripts (no LLM calls) + write a manual-check sheet
python audit_outputs.py --input path/to/series.zip --run outputs/promos_raw.json --manual-sheet eval/manual_check.csv

# Unit + end-to-end tests with a fake LLM (no API key needed)
pip install -r requirements-dev.txt
pytest -q

# Try the tool on a show it has never seen (original 3-episode corporate drama, same format)
python samples/make_fake_show.py
python run_cli.py --input samples/fake_show_boardroom --provider nvidia --out outputs_fake_show
```

`eval/manual_check.csv` has the tool's claims next to the script proof and empty `HUMAN_` Y/N columns. The tool never fills those in — a person does. (`eval/*.csv` is git-ignored because it quotes script lines.)

---

## v5 Fixes (found by re-checking the v4 output)

Found by re-checking the v4 output line by line against the scripts:

| Problem in v4 | Evidence | Fix on this branch |
|---|---|---|
| Overlays cross-wired between promos | #12 (Ep 33) overlay says "cathedral nave"; Ep 33 never mentions a cathedral | Overlays written per promo from its own clip (Stage 4C) |
| Overlay reversed who threatens whom | #1 overlay "Husband declares lethal intent on bride"; the line is Alexander saying villains plan to kill *his* wife | Overlay prompt must state who-does-what-to-whom first. **Not detectable by code** — still needs a human |
| "Verified cliffhanger" describes an earlier beat, not the cut point | 7 of 20 promos (#4, #8, #11, #12, #13, #15, #16) per `audit_outputs.py`; I checked #4, #11, #13, #15, #16 by hand | Flagged automatically as `NEEDS REVIEW` in every report/CSV |
| Prompts contained this show's plot ("Duke reveal", "Ep 1-10 altar betrayal", "scarred beggar") | `prompts.py` stitcher + ranker examples | Examples made genre-neutral; tested parsing + pipeline plumbing on an original fake show |
| Cached runs printed "API calls: 0" | `llm_client.py` only counts real calls | Report shows cache reuse **and** the original fresh-run usage |
| Same model scored its own picks | Scout, critic, ranker were all Nemotron 30B | Optional `--critic-provider` for an independent judge |

### What v5 still gets wrong (checked against the scripts)

v5 full run: 50 episodes, 38 API calls, 342K tokens, **14 promos ranked, 15 rejected, 4 auto-flagged**. Run-to-run variation means v5 is not directly comparable with v4 (the scout picks different scenes each run; one scout batch failed over from NVIDIA to Gemini).

| Problem | Evidence |
|---|---|
| Overlay still reverses meaning — the who-does-what step did **not** fix this | #14 says "Lady Juliana refuses dying boy"; at 35.9s she says "Let me grab my ledger. We go now." #7 says "Alexander assaults Juliana"; he finds a wound someone else caused and asks "Who did this?" |
| Clips cut just before their best line | #5 ends at 41.5s; "They declared me dead." starts at 41.6s. #3 ends 0.3s before the High Priest's "…lose her twice" |
| Two promos reuse the same footage | #6 and #13 share 20.4s of Episode 7 (00:50.8–01:11.2); the overlap filter only covers stitched promos |
| Critic scores are lenient | #7 opens on a 4s wide forest shot (Hook 4/5); #6 ends after the conflict is resolved (Cliffhanger 5/5) |
| One auto-flag is a false alarm | #1 "at altar": the scene is at the altar, but the clip never shows it |

**Next fixes (not built):** use a quoted line from the clip as the overlay instead of generated text; extend a cut when a dialogue line starts within 1s after it; block overlapping continuous promos; a human-labelled gold set. Every promo still needs human approval before posting.

---

## Repository Structure

```text
microdrama-promo-engine/
├── parser.py              # Regex screenplay parser & Beat ID normalizer
├── llm_client.py          # Multi-provider OpenAI-SDK wrapper with 45s timeout & failover
├── prompts.py             # Scout, Trimmer, Critic, Stitcher, Ranker, Overlay prompts
├── pipeline.py            # Stages 1–4C orchestration & exporters
├── grounding.py           # Deterministic grounding checks (no LLM)
├── audit_outputs.py       # Offline audit of any run + manual-check sheet
├── run_cli.py             # Command-line runner
├── app.py                 # Streamlit UI + CSV/Markdown/JSON downloads
├── parser_diagnostics.py  # Prints per-episode parse stats (was test_parser.py)
├── samples/               # Original fake show for testing on an unseen show
├── tests/                 # pytest: parser, grounding, end-to-end with a fake LLM
├── outputs/               # v4 run on the full pack (unedited)
│   ├── ranked_promos.csv
│   ├── promo_report.md
│   ├── promos_raw.json
│   └── Solution_Doc.pdf
├── outputs_v5/            # v5 run on the full pack (unedited)
├── requirements.txt / requirements-dev.txt
├── .env.example
└── .gitignore
```
