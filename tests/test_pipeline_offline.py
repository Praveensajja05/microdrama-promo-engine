"""End-to-end pipeline test with a FAKE LLM: no API keys, no network.

It checks the plumbing (stages, validators, overlays per promo, review flags, cache metrics),
NOT promo quality. Quality needs a real model + a human check.
"""

import json
import re
from pathlib import Path

from parser import load_episodes_from_path
from pipeline import PromoPipeline, export_outputs
from prompts import (
    CRITIC_SYSTEM_PROMPT,
    OVERLAY_SYSTEM_PROMPT,
    RANKER_SYSTEM_PROMPT,
    SCOUT_SYSTEM_PROMPT,
    STITCHER_SYSTEM_PROMPT,
    TRIMMER_SYSTEM_PROMPT,
)

FAKE_SHOW = Path(__file__).resolve().parent.parent / "samples" / "fake_show_boardroom"
BEAT_RE = re.compile(r"\[(E(\d+)_B\d+) \| ([\d.]+)s-([\d.]+)s\]")


class FakeLLM:
    """Mimics SwappableLLMClient's interface and returns canned, schema-valid JSON."""

    def __init__(self, provider="fake", model="fake-scout", strong_model="fake-strong", overlay_text=None):
        self.provider, self.model, self.strong_model = provider, model, strong_model
        self.total_calls = self.total_prompt_tokens = self.total_completion_tokens = 0
        self.failover_events = []
        self.calls_by_prompt = {}
        self.overlay_text = overlay_text

    def call_json(self, system_prompt, user_prompt, model_override=None, **_):
        self.total_calls += 1
        self.total_prompt_tokens += 100
        self.total_completion_tokens += 20
        name = {
            SCOUT_SYSTEM_PROMPT: "scout", TRIMMER_SYSTEM_PROMPT: "trimmer", CRITIC_SYSTEM_PROMPT: "critic",
            STITCHER_SYSTEM_PROMPT: "stitcher", RANKER_SYSTEM_PROMPT: "ranker", OVERLAY_SYSTEM_PROMPT: "overlay",
        }[system_prompt]
        self.calls_by_prompt[name] = self.calls_by_prompt.get(name, 0) + 1
        return getattr(self, "_" + name)(user_prompt)

    @staticmethod
    def _beats_by_episode(text):
        eps = {}
        for bid, ep, s, e in BEAT_RE.findall(text):
            eps.setdefault(int(ep), []).append((bid, float(s), float(e)))
        return eps

    def _scout(self, user_prompt):
        cands, atoms = [], []
        for ep, beats in self._beats_by_episode(user_prompt).items():
            inside = [b for b in beats if b[2] <= 35.0]           # ~0-35s continuous promo
            cands.append({
                "title": f"Episode {ep} opener", "trope": "Betrayal",
                "segments": [{"episode": ep, "start_beat_id": inside[0][0], "end_beat_id": inside[-1][0]}],
                "hook_0_3s": "x", "cliffhanger": "x", "why_it_works": "x", "scout_score": 8,
            })
            late = [b for b in beats if b[1] >= 44.0 and b[2] <= 58.0]  # ~10-14s atom
            if late:
                atoms.append({"role": "FLASH_FORWARD_CLIMAX", "episode": ep,
                              "start_beat_id": late[0][0], "end_beat_id": late[-1][0], "summary": "late moment"})
        return {"updated_story_summary": "fake", "candidates": cands, "stitch_atoms": atoms}

    def _trimmer(self, user_prompt):
        return {"trimmed": []}

    def _critic(self, user_prompt):
        reviews = []
        for block in user_prompt.split("=== CANDIDATE ")[1:]:
            cid = block.split(" ", 1)[0]
            lines = [l for l in block.splitlines() if l.startswith("[")]
            first = lines[0].split("] ", 1)[1]
            last = lines[-1].split("] ", 1)[1]
            reviews.append({
                "candidate_id": cid, "verified_title": f"Clip {cid}",
                "verified_hook_0_3s": first, "verified_cliffhanger": last, "verified_why_it_works": "x",
                "hook_3s": 5, "standalone_clarity": 4, "cliffhanger_pull": 4, "verdict": "PASS", "critique": "ok",
            })
        return {"reviews": reviews}

    def _stitcher(self, user_prompt):
        atoms = json.loads(user_prompt)["stitch_atoms"]
        if len(atoms) < 2:
            return {"stitched_promos": []}
        a, b = atoms[0], atoms[1]
        return {"stitched_promos": [{
            "id": "ST01", "title": "Flash forward", "trope": "Flash-Forward Mashup",
            "segments": [{k: a[k] for k in ("episode", "start_beat_id", "end_beat_id")},
                         {k: b[k] for k in ("episode", "start_beat_id", "end_beat_id")}],
            "hook_0_3s": "x", "cliffhanger": "x", "why_it_works": "x",
        }]}

    def _ranker(self, user_prompt):
        pool = json.loads(user_prompt)["verified_promos"]
        return {"ranked_ids": [
            {"id": p["id"], "title": p["title"], "rank": i + 1, "posting_tier": "Wave 1 (Launch Hero)",
             "overlay_hook": "THIS SHOULD BE IGNORED", "ranking_reason": "fake"}
            for i, p in enumerate(pool)
        ]}

    def _overlay(self, user_prompt):
        return {"who_does_what_to_whom": "fake", "overlay_hook": self.overlay_text or "Heiress fights back"}


def _run(tmp_path, llm, critic_llm=None, use_cache=True):
    eps = load_episodes_from_path(str(FAKE_SHOW))
    pipe = PromoPipeline(eps, llm, strong_model=llm.strong_model, cache_dir=str(tmp_path / "cache"), critic_llm=critic_llm)
    return pipe.run(use_cache=use_cache)


def test_full_pipeline_runs_on_unseen_show(tmp_path):
    llm = FakeLLM()
    res = _run(tmp_path, llm)
    promos = res["ranked_promos"]
    assert len(promos) >= 3
    for p in promos:
        assert 20.0 <= p["total_duration_sec"] <= 60.0
        assert "review_flags" in p
    # overlays come from the per-promo writer, never from the batch ranker
    assert all(p["overlay_hook"] == "Heiress fights back" for p in promos)
    assert llm.calls_by_prompt["overlay"] == len(promos)
    assert res["meta"]["cache_hits"] == 0


def test_overlay_naming_another_episodes_place_is_flagged(tmp_path):
    # "penthouse" only exists in Episode 2, so Episode 1 and 3 promos must be flagged.
    res = _run(tmp_path, FakeLLM(overlay_text="Heiress strikes back at the penthouse"))
    flagged = {tuple(s["episode"] for s in p["segments"]): p["review_flags"] for p in res["ranked_promos"]}
    assert any("penthouse" in f for f in flagged[(1,)])
    assert not any("penthouse" in f for f in flagged[(2,)])
    assert res["meta"]["promos_flagged_for_review"] >= 2


def test_cached_rerun_reports_original_usage(tmp_path):
    first = _run(tmp_path, FakeLLM())
    fresh_calls = first["meta"]["llm_calls"]
    assert fresh_calls > 0

    second_llm = FakeLLM()
    second = _run(tmp_path, second_llm)
    m = second["meta"]
    assert m["llm_calls"] == 0 and m["cache_hits"] > 0
    assert m["fresh_run_usage"]["llm_calls"] == fresh_calls

    out = export_outputs(second, str(tmp_path / "out"))
    header = (out / "promo_report.md").read_text(encoding="utf-8")
    assert f"the original fresh run used {fresh_calls} calls" in header


def test_independent_critic_is_used_for_scoring(tmp_path):
    main, judge = FakeLLM(), FakeLLM(provider="judge", strong_model="judge-model")
    res = _run(tmp_path, main, critic_llm=judge)
    assert judge.calls_by_prompt.get("critic", 0) >= 1
    assert "critic" not in main.calls_by_prompt
    assert res["meta"]["critic_provider"] == "judge"
    assert res["meta"]["llm_calls"] == main.total_calls + judge.total_calls
