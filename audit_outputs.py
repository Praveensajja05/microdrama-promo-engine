"""Audit an existing run's promos without calling any LLM.

Usage:
  python audit_outputs.py --input data/series.zip --run outputs/promos_raw.json
  python audit_outputs.py --input data/series.zip --run outputs/promos_raw.json --manual-sheet eval/manual_check.csv

1. Runs the deterministic grounding checks (grounding.py) on every ranked promo.
2. Optionally writes a manual-check sheet: one row per promo with the script proof
   next to the tool's claims, plus empty Y/N columns for a HUMAN to fill in.
   The tool never fills those columns itself.
"""

import argparse
import csv
import json
from pathlib import Path

from grounding import audit_promo, character_words, place_and_people_words
from parser import load_episodes_from_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline grounding audit of a promo run")
    ap.add_argument("--input", required=True, help="Same screenplay input the run used (.zip / folder / .txt)")
    ap.add_argument("--run", default="outputs/promos_raw.json")
    ap.add_argument("--manual-sheet", default=None, help="Write a CSV for manual Y/N checking")
    args = ap.parse_args()

    episodes = load_episodes_from_path(args.input)
    specific = place_and_people_words(episodes)
    names = character_words(episodes)
    result = json.loads(Path(args.run).read_text(encoding="utf-8"))
    promos = result.get("ranked_promos", [])

    flagged = 0
    rows = []
    for p in promos:
        flags = audit_promo(p, specific, names, {e.episode_num: e for e in episodes})
        if flags:
            flagged += 1
        segs = " + ".join(f"Ep {s['episode']} [{s['start_ts']} - {s['end_ts']}]" for s in p.get("segments", []))
        print(f"#{p.get('rank')} {p.get('candidate_id')} {segs}")
        for f in flags:
            print(f"    FLAG: {f}")
        rows.append({
            "rank": p.get("rank"),
            "id": p.get("candidate_id"),
            "segments": segs,
            "duration_sec": p.get("total_duration_sec"),
            "tool_overlay": p.get("overlay_hook", ""),
            "tool_hook": p.get("hook_0_3s", ""),
            "script_first_3s": " / ".join(p["segments"][0].get("opening_3s", [])) if p.get("segments") else "",
            "tool_cliffhanger": p.get("cliffhanger", ""),
            "script_last_3s": " / ".join(p["segments"][-1].get("closing_3s", [])) if p.get("segments") else "",
            "auto_flags": " | ".join(flags),
            "HUMAN_hook_in_first_3s (Y/N)": "",
            "HUMAN_ends_unresolved (Y/N)": "",
            "HUMAN_makes_sense_to_stranger (Y/N)": "",
            "HUMAN_overlay_correct (Y/N)": "",
            "HUMAN_would_post (Y/N)": "",
            "HUMAN_notes": "",
        })

    print(f"\n{flagged} of {len(promos)} ranked promos have at least one automatic flag.")
    print("Flags mean 'a human should look', not 'definitely wrong'. Reversed meanings are NOT detectable here.")

    if args.manual_sheet:
        out = Path(args.manual_sheet)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["rank"])
            w.writeheader()
            w.writerows(rows)
        print(f"Manual-check sheet written to {out} (fill the HUMAN_ columns yourself).")


if __name__ == "__main__":
    main()
