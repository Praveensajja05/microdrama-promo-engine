import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from grounding import audit_promo, character_words, place_and_people_words
from llm_client import SwappableLLMClient
from parser import Episode, format_timestamp
from prompts import (
    CRITIC_SYSTEM_PROMPT,
    OVERLAY_SYSTEM_PROMPT,
    RANKER_SYSTEM_PROMPT,
    SCOUT_SYSTEM_PROMPT,
    STITCHER_SYSTEM_PROMPT,
    TRIMMER_SYSTEM_PROMPT,
)

MIN_PROMO_SEC = 20.0
MAX_PROMO_SEC = 60.0
MAX_TRIMMABLE_SEC = 220.0
MIN_INTRA_SEG_SEC = 8.0
MIN_INTRA_GAP_SEC = 6.0
MIN_CRITIC_SINGLE_SCORE = 3
MIN_CRITIC_TOTAL_SCORE = 11


class PromoPipeline:
    def __init__(
        self,
        episodes: List[Episode],
        llm: SwappableLLMClient,
        strong_model: Optional[str] = None,
        batch_size: int = 5,
        cache_dir: str = ".cache",
        critic_llm: Optional[SwappableLLMClient] = None,
    ):
        self.episodes = episodes
        self.llm = llm
        self.strong_model = strong_model or llm.model
        # Optional independent judge: a DIFFERENT model/provider scores the clips, so the
        # scout model is not grading its own picks. Falls back to the primary client.
        self.critic_llm = critic_llm
        self.critic_model = critic_llm.strong_model if critic_llm else self.strong_model
        self.cache_hits = 0
        self.specific_words = place_and_people_words(episodes)
        self.character_names = character_words(episodes)
        self.batch_size = batch_size
        self.ep_map: Dict[int, Episode] = {ep.episode_num: ep for ep in episodes}
        self.beat_index: Dict[str, Tuple[int, int]] = {}
        for ep in episodes:
            for idx, b in enumerate(ep.beats):
                self.beat_index[b.beat_id] = (ep.episode_num, idx)

        prompt_hash = hashlib.md5(
            (
                SCOUT_SYSTEM_PROMPT
                + TRIMMER_SYSTEM_PROMPT
                + CRITIC_SYSTEM_PROMPT
                + STITCHER_SYSTEM_PROMPT
                + RANKER_SYSTEM_PROMPT
                + OVERLAY_SYSTEM_PROMPT
            ).encode()
        ).hexdigest()[:6]
        fingerprint = hashlib.md5(
            f"v5:{llm.provider}:{llm.model}:{self.strong_model}:{self.critic_model}:{len(episodes)}:{sum(len(e.beats) for e in episodes)}:{prompt_hash}".encode()
        ).hexdigest()[:10]
        self.cache_path = Path(cache_dir) / f"run_{fingerprint}"
        self.cache_path.mkdir(parents=True, exist_ok=True)

    # ---------- Cache Helpers ----------
    def _cache_load(self, cache_file: Path, use_cache: bool) -> Optional[Any]:
        """Returns cached JSON (and counts the hit) or None."""
        if use_cache and cache_file.exists():
            self.cache_hits += 1
            return json.loads(cache_file.read_text(encoding="utf-8"))
        return None

    def _usage(self) -> Dict[str, int]:
        clients = [self.llm] + ([self.critic_llm] if self.critic_llm else [])
        return {
            "llm_calls": sum(c.total_calls for c in clients),
            "prompt_tokens": sum(c.total_prompt_tokens for c in clients),
            "completion_tokens": sum(c.total_completion_tokens for c in clients),
        }

    # ---------- Deterministic Helpers ----------
    def _resolve_segment(
        self,
        seg: Dict[str, Any],
        snap_short: bool = False,
        snap_long: bool = False,
        expand_bounds: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        s_id = str(seg.get("start_beat_id", "")).strip()
        e_id = str(seg.get("end_beat_id", "")).strip()

        if s_id not in self.beat_index or e_id not in self.beat_index:
            return None, f"HALLUCINATED_BEAT_ID ({s_id} -> {e_id})"

        ep_s, idx_s = self.beat_index[s_id]
        ep_e, idx_e = self.beat_index[e_id]
        if ep_s != ep_e:
            return None, f"CROSS_EPISODE_SINGLE_SEGMENT ({s_id} -> {e_id})"
        if idx_e < idx_s:
            idx_s, idx_e = idx_e, idx_s

        beats = self.ep_map[ep_s].beats
        start_sec = beats[idx_s].start_sec
        end_sec = max(b.end_sec for b in beats[idx_s : idx_e + 1])
        dur = round(end_sec - start_sec, 2)
        snapped = False

        # Deterministic under-trim recovery inside an oversized scene's original bounds
        if expand_bounds and 4.0 <= dur < MIN_PROMO_SEC:
            min_idx, max_idx = expand_bounds
            target_sec = 25.0
            while dur < target_sec and idx_e + 1 <= max_idx:
                next_end = max(end_sec, beats[idx_e + 1].end_sec)
                if next_end - start_sec > 56.0:
                    break
                idx_e += 1
                end_sec = next_end
                dur = round(end_sec - start_sec, 2)
                snapped = True
            while dur < MIN_PROMO_SEC and idx_s - 1 >= min_idx:
                next_start = beats[idx_s - 1].start_sec
                if end_sec - next_start > 56.0:
                    break
                idx_s -= 1
                start_sec = next_start
                dur = round(end_sec - start_sec, 2)
                snapped = True

        if snap_short and 15.0 <= dur < MIN_PROMO_SEC:
            while dur < MIN_PROMO_SEC and idx_e + 1 < len(beats):
                idx_e += 1
                end_sec = max(b.end_sec for b in beats[idx_s : idx_e + 1])
                dur = round(end_sec - start_sec, 2)
                snapped = True

        if snap_long and MAX_PROMO_SEC < dur <= 66.0:
            while dur > MAX_PROMO_SEC and idx_e > idx_s:
                idx_e -= 1
                end_sec = max(b.end_sec for b in beats[idx_s : idx_e + 1])
                dur = round(end_sec - start_sec, 2)
                snapped = True

        slice_beats = beats[idx_s : idx_e + 1]
        opening_3s = [b.to_compact_line() for b in slice_beats if b.start_sec <= start_sec + 3.0]
        closing_3s = [b.to_compact_line() for b in slice_beats if b.end_sec >= end_sec - 3.0]

        return {
            "episode": ep_s,
            "start_beat_id": beats[idx_s].beat_id,
            "end_beat_id": beats[idx_e].beat_id,
            "idx_s": idx_s,
            "idx_e": idx_e,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "start_ts": format_timestamp(start_sec),
            "end_ts": format_timestamp(end_sec),
            "duration_sec": dur,
            "auto_snapped": snapped,
            "opening_3s": opening_3s[:4],
            "closing_3s": closing_3s[-4:],
            "transcript": "\n".join(b.to_compact_line() for b in slice_beats),
        }, None

    def _validate_promo(
        self,
        raw_promo: Dict[str, Any],
        cid: str,
        route_oversized: bool = False,
        snap_long: bool = False,
        expand_bounds: Optional[Tuple[int, int]] = None,
        require_multi_episode: bool = False,
    ) -> Tuple[str, Dict[str, Any]]:
        """Returns (status, payload) where status in {'VALID', 'OVERSIZED', 'REJECTED'}."""
        raw_segs = raw_promo.get("segments", [])
        if not raw_segs:
            return "REJECTED", {"candidate_id": cid, "title": raw_promo.get("title", ""), "stage": "validator", "reason": "NO_SEGMENTS"}

        if len(raw_segs) > 2:
            raw_segs = raw_segs[:2]

        resolved_segs = []
        is_single = len(raw_segs) == 1
        for seg in raw_segs:
            res, err = self._resolve_segment(
                seg,
                snap_short=is_single,
                snap_long=(is_single and snap_long),
                expand_bounds=(expand_bounds if is_single else None),
            )
            if err:
                return "REJECTED", {"candidate_id": cid, "title": raw_promo.get("title", ""), "stage": "validator", "reason": err}
            if res["duration_sec"] < 3.0:
                return "REJECTED", {"candidate_id": cid, "title": raw_promo.get("title", ""), "stage": "validator", "reason": f"SEGMENT_TOO_SHORT ({res['duration_sec']}s)"}
            resolved_segs.append(res)

        distinct_eps = {s["episode"] for s in resolved_segs}
        if require_multi_episode and len(distinct_eps) < 2:
            return "REJECTED", {"candidate_id": cid, "title": raw_promo.get("title", ""), "stage": "stitched_validator", "reason": "NOT_MULTI_EPISODE"}

        # Guardrails for Intra-Episode Compression Cuts (2 segments in same episode)
        if len(resolved_segs) == 2 and len(distinct_eps) == 1:
            a, b = resolved_segs[0], resolved_segs[1]
            if a["end_sec"] > b["start_sec"]:
                return "REJECTED", {
                    "candidate_id": cid,
                    "title": raw_promo.get("title", ""),
                    "stage": "validator",
                    "reason": "OVERLAPPING_OR_OUT_OF_ORDER_INTRA_CUT",
                }
            gap_sec = round(b["start_sec"] - a["end_sec"], 2)
            # If skipped gap is tiny (<6s) and merged span fits <=60s, merge into a clean Continuous cut
            if gap_sec < MIN_INTRA_GAP_SEC and (b["end_sec"] - a["start_sec"]) <= MAX_PROMO_SEC:
                merged, err = self._resolve_segment(
                    {"start_beat_id": a["start_beat_id"], "end_beat_id": b["end_beat_id"]},
                    snap_short=True,
                    snap_long=snap_long,
                )
                if not err and merged:
                    resolved_segs = [merged]
            else:
                if a["duration_sec"] < MIN_INTRA_SEG_SEC or b["duration_sec"] < MIN_INTRA_SEG_SEC:
                    return "REJECTED", {
                        "candidate_id": cid,
                        "title": raw_promo.get("title", ""),
                        "stage": "validator",
                        "reason": f"INTRA_SEGMENT_TOO_SHORT ({a['duration_sec']}s + {b['duration_sec']}s; min {MIN_INTRA_SEG_SEC}s each)",
                    }
                # Gentle top-end snap if 2-part intra cut ran slightly over 60s (up to 68s)
                combined_dur = round(a["duration_sec"] + b["duration_sec"], 2)
                if snap_long and MAX_PROMO_SEC < combined_dur <= 68.0:
                    beats = self.ep_map[b["episode"]].beats
                    idx_s2, idx_e2 = b["idx_s"], b["idx_e"]
                    while combined_dur > MAX_PROMO_SEC and idx_e2 > idx_s2:
                        idx_e2 -= 1
                        new_end = max(bt.end_sec for bt in beats[idx_s2 : idx_e2 + 1])
                        new_b_dur = round(new_end - b["start_sec"], 2)
                        if new_b_dur < MIN_INTRA_SEG_SEC:
                            break
                        combined_dur = round(a["duration_sec"] + new_b_dur, 2)
                    trimmed_b, err2 = self._resolve_segment(
                        {"start_beat_id": beats[idx_s2].beat_id, "end_beat_id": beats[idx_e2].beat_id}
                    )
                    if not err2 and trimmed_b:
                        resolved_segs[1] = trimmed_b

        # Block overlapping segments inside same episode (defensive check)
        for i in range(len(resolved_segs)):
            for j in range(i + 1, len(resolved_segs)):
                a, b = resolved_segs[i], resolved_segs[j]
                if a["episode"] == b["episode"] and max(a["start_sec"], b["start_sec"]) < min(a["end_sec"], b["end_sec"]):
                    return "REJECTED", {"candidate_id": cid, "title": raw_promo.get("title", ""), "stage": "validator", "reason": "OVERLAPPING_STITCH_SEGMENTS"}

        total_dur = round(sum(s["duration_sec"] for s in resolved_segs), 2)

        if route_oversized and len(resolved_segs) == 1 and MAX_PROMO_SEC < total_dur <= MAX_TRIMMABLE_SEC:
            return "OVERSIZED", {
                "candidate_id": cid,
                "raw_promo": raw_promo,
                "segment": resolved_segs[0],
                "original_duration_sec": total_dur,
            }

        if not (MIN_PROMO_SEC <= total_dur <= MAX_PROMO_SEC):
            return "REJECTED", {
                "candidate_id": cid,
                "title": raw_promo.get("title", ""),
                "stage": "validator",
                "reason": f"DURATION_OUT_OF_BOUNDS ({total_dur}s not in 20-60s)",
                "segments_preview": [f"Ep{s['episode']} {s['start_ts']}-{s['end_ts']}" for s in resolved_segs],
            }

        distinct_eps_final = {s["episode"] for s in resolved_segs}
        if len(resolved_segs) == 1:
            promo_format = "Continuous"
        elif len(distinct_eps_final) == 1:
            promo_format = "Intra-Episode Cut"
        else:
            promo_format = "Cross-Episode Stitched"

        clean_segs = [{k: v for k, v in s.items() if k not in {"idx_s", "idx_e"}} for s in resolved_segs]
        return "VALID", {
            "candidate_id": cid,
            "title": raw_promo.get("title", "Untitled Promo"),
            "trope": raw_promo.get("trope", "Drama"),
            "promo_format": promo_format,
            "is_stitched": len(clean_segs) > 1,
            "total_duration_sec": total_dur,
            "segments": clean_segs,
            "hook_0_3s": raw_promo.get("hook_0_3s", ""),
            "cliffhanger": raw_promo.get("cliffhanger", ""),
            "why_it_works": raw_promo.get("why_it_works", ""),
            "scout_score": raw_promo.get("scout_score", 7),
        }

    @staticmethod
    def _seg_overlap_ratio(a: Dict[str, Any], b: Dict[str, Any]) -> float:
        if a["episode"] != b["episode"]:
            return 0.0
        inter = max(0.0, min(a["end_sec"], b["end_sec"]) - max(a["start_sec"], b["start_sec"]))
        min_len = min(a["duration_sec"], b["duration_sec"])
        return (inter / min_len) if min_len > 0 else 0.0

    @classmethod
    def _promo_overlap_ratio(cls, cand_a: Dict[str, Any], cand_b: Dict[str, Any]) -> float:
        """Fraction of cand_a's total duration overlapped by cand_b's segments."""
        total_a = cand_a.get("total_duration_sec", 0.0)
        if total_a <= 0:
            return 0.0
        overlap_sec = 0.0
        for sa in cand_a["segments"]:
            for sb in cand_b["segments"]:
                if sa["episode"] == sb["episode"]:
                    overlap_sec += max(0.0, min(sa["end_sec"], sb["end_sec"]) - max(sa["start_sec"], sb["start_sec"]))
        return overlap_sec / total_a

    @classmethod
    def _suppress_overlaps(cls, candidates: List[Dict[str, Any]], rejected: List[Dict[str, Any]], threshold: float = 0.55) -> List[Dict[str, Any]]:
        ordered = sorted(candidates, key=lambda c: c.get("scout_score", 0), reverse=True)
        kept: List[Dict[str, Any]] = []
        for cand in ordered:
            if cand.get("promo_format") == "Cross-Episode Stitched":
                kept.append(cand)
                continue
            duplicate_of = None
            for k in kept:
                if k.get("promo_format") == "Cross-Episode Stitched":
                    continue
                if cls._promo_overlap_ratio(cand, k) >= threshold or cls._promo_overlap_ratio(k, cand) >= threshold:
                    duplicate_of = k["candidate_id"]
                    break
            if duplicate_of:
                rejected.append({
                    "candidate_id": cand["candidate_id"],
                    "title": cand["title"],
                    "stage": "overlap_suppression",
                    "reason": f"OVERLAPS_WITH_{duplicate_of}",
                })
            else:
                kept.append(cand)
        return kept

    @staticmethod
    def _evaluate_critic_gate(rev: Dict[str, Any]) -> Tuple[bool, int, str]:
        """Deterministic math gate so LLM 'PASS' can never override failing scores."""
        if not rev:
            return False, 0, "MISSING_CRITIC_REVIEW"
        try:
            h = int(rev.get("hook_3s", 0))
            cl = int(rev.get("standalone_clarity", 0))
            cf = int(rev.get("cliffhanger_pull", 0))
        except (TypeError, ValueError):
            return False, 0, "INVALID_CRITIC_SCORES"

        total = h + cl + cf
        llm_verdict = str(rev.get("verdict", "REJECT")).strip().upper()
        critique = str(rev.get("critique", "")).strip()

        if min(h, cl, cf) < MIN_CRITIC_SINGLE_SCORE or total < MIN_CRITIC_TOTAL_SCORE:
            reason = f"[Deterministic Gate: {h}/{cl}/{cf} = {total}/15 below threshold] {critique}"
            return False, total, reason
        if llm_verdict == "REJECT":
            return False, total, critique or "Rejected by Cold-Scroll Critic"
        return True, total, critique

    def _run_critic_batch(self, clips: List[Dict[str, Any]], cache_prefix: str, use_cache: bool) -> Dict[str, Dict[str, Any]]:
        reviews_map: Dict[str, Dict[str, Any]] = {}
        chunk_size = 5
        for start in range(0, len(clips), chunk_size):
            chunk = clips[start : start + chunk_size]
            c_idx = (start // chunk_size) + 1
            cache_file = self.cache_path / f"{cache_prefix}_{c_idx:02d}.json"

            cres = self._cache_load(cache_file, use_cache)
            if cres is None:
                isolated = []
                for c in chunk:
                    clip_text = "\n--- SCENE CUT ---\n".join(s["transcript"] for s in c["segments"])
                    isolated.append(
                        f"=== CANDIDATE {c['candidate_id']} | Format: {c.get('promo_format', 'Continuous')} "
                        f"(Total Duration: {c['total_duration_sec']}s) ===\n{clip_text}"
                    )
                judge = self.critic_llm or self.llm
                cres = judge.call_json(
                    CRITIC_SYSTEM_PROMPT,
                    "Review these isolated cold-scroll clips with ZERO external context:\n\n" + "\n\n".join(isolated),
                    model_override=self.critic_model,
                )
                cache_file.write_text(json.dumps(cres, indent=2), encoding="utf-8")

            for rev in cres.get("reviews", []):
                cid = str(rev.get("candidate_id", "")).strip().upper()
                if cid:
                    reviews_map[cid] = rev
        return reviews_map

    # ---------- Main Execution ----------
    def run(self, progress_cb: Optional[Callable[[str, float], None]] = None, use_cache: bool = True) -> Dict[str, Any]:
        def report(msg: str, frac: float):
            if progress_cb:
                progress_cb(msg, frac)

        # STAGE 1: Batched Map (Episode Scout)
        all_raw_candidates: List[Dict[str, Any]] = []
        all_stitch_atoms: List[Dict[str, Any]] = []
        rolling_summary = "Beginning of series."

        batches = [self.episodes[i : i + self.batch_size] for i in range(0, len(self.episodes), self.batch_size)]
        for b_idx, batch in enumerate(batches, start=1):
            ep_range = f"Ep {batch[0].episode_num}-{batch[-1].episode_num}"
            report(f"Stage 1/5 — Scouting {ep_range} (batch {b_idx}/{len(batches)})...", 0.05 + 0.35 * (b_idx / len(batches)))

            cache_file = self.cache_path / f"scout_batch_{b_idx:02d}.json"
            res = self._cache_load(cache_file, use_cache)
            if res is None:
                script_blob = "\n\n".join(ep.to_prompt_text() for ep in batch)
                user_prompt = (
                    f"STORY SO FAR:\n{rolling_summary}\n\n"
                    f"EPISODES IN THIS BATCH ({ep_range}):\n{script_blob}\n\n"
                    "Return JSON only."
                )
                res = self.llm.call_json(SCOUT_SYSTEM_PROMPT, user_prompt)
                cache_file.write_text(json.dumps(res, indent=2), encoding="utf-8")

            rolling_summary = res.get("updated_story_summary", rolling_summary)
            all_raw_candidates.extend(res.get("candidates", []))
            all_stitch_atoms.extend(res.get("stitch_atoms", []))

        # STAGE 2: Deterministic Initial Validation (Route 60-220s scenes to Smart Trimmer)
        report("Stage 2/5 — Validating Beat IDs & routing oversized scenes to Smart Trimmer...", 0.45)
        valid_candidates: List[Dict[str, Any]] = []
        oversized_queue: List[Dict[str, Any]] = []
        rejected_log: List[Dict[str, Any]] = []

        for idx, raw_c in enumerate(all_raw_candidates, start=1):
            cid = f"C{idx:02d}"
            status, payload = self._validate_promo(raw_c, cid, route_oversized=True, snap_long=False)
            if status == "VALID":
                valid_candidates.append(payload)
            elif status == "OVERSIZED":
                oversized_queue.append(payload)
            else:
                rejected_log.append(payload)

        # STAGE 2.5: Surgical Smart Trimmer (Continuous OR Intra-Episode Compression Cut)
        recovered_count = 0
        if oversized_queue:
            report(
                f"Stage 2.5/5 — Trimming & compressing {len(oversized_queue)} oversized flagship scenes into 25-55s cuts...",
                0.55,
            )
            trim_chunk_size = 4
            trim_results: Dict[str, Dict[str, Any]] = {}

            for t_start in range(0, len(oversized_queue), trim_chunk_size):
                chunk = oversized_queue[t_start : t_start + trim_chunk_size]
                t_idx = (t_start // trim_chunk_size) + 1
                cache_file = self.cache_path / f"trimmer_chunk_{t_idx:02d}.json"

                tres = self._cache_load(cache_file, use_cache)
                if tres is None:
                    blocks = []
                    for item in chunk:
                        seg = item["segment"]
                        blocks.append(
                            f"=== {item['candidate_id']} | Episode {seg['episode']} | Title: {item['raw_promo'].get('title', '')} "
                            f"| Currently {item['original_duration_sec']}s (TOO LONG — carve 1 continuous 25-55s cut OR 2-part Intra-Episode Cut) ===\n"
                            f"{seg['transcript']}"
                        )
                    tres = self.llm.call_json(
                        TRIMMER_SYSTEM_PROMPT,
                        "Trim or compress each oversized scene into a tight 25-55s promo:\n\n" + "\n\n".join(blocks),
                    )
                    cache_file.write_text(json.dumps(tres, indent=2), encoding="utf-8")

                for tr in tres.get("trimmed", []):
                    trim_results[str(tr.get("candidate_id", "")).strip().upper()] = tr

            for item in oversized_queue:
                cid = item["candidate_id"]
                tr = trim_results.get(cid)
                if not tr:
                    rejected_log.append({
                        "candidate_id": cid,
                        "title": item["raw_promo"].get("title", ""),
                        "stage": "smart_trimmer",
                        "reason": f"TRIMMER_OMITTED (original {item['original_duration_sec']}s)",
                    })
                    continue

                # Accept either v4 `segments` list OR legacy flat `start_beat_id`/`end_beat_id`
                raw_tr_segs = tr.get("segments")
                if isinstance(raw_tr_segs, list) and raw_tr_segs:
                    new_segs = [
                        {
                            "episode": item["segment"]["episode"],
                            "start_beat_id": s.get("start_beat_id", ""),
                            "end_beat_id": s.get("end_beat_id", ""),
                        }
                        for s in raw_tr_segs
                        if isinstance(s, dict)
                    ]
                else:
                    new_segs = [{
                        "episode": item["segment"]["episode"],
                        "start_beat_id": tr.get("start_beat_id", ""),
                        "end_beat_id": tr.get("end_beat_id", ""),
                    }]

                patched_raw = dict(item["raw_promo"])
                patched_raw["segments"] = new_segs
                orig_bounds = (item["segment"]["idx_s"], item["segment"]["idx_e"])
                status, payload = self._validate_promo(
                    patched_raw,
                    cid,
                    route_oversized=False,
                    snap_long=True,
                    expand_bounds=orig_bounds,
                )
                if status == "VALID":
                    payload["trimmed_from_sec"] = item["original_duration_sec"]
                    payload["trim_note"] = tr.get("trim_note", "")
                    valid_candidates.append(payload)
                    recovered_count += 1
                else:
                    payload["stage"] = "trimmer_validator"
                    rejected_log.append(payload)

        valid_candidates = self._suppress_overlaps(valid_candidates, rejected_log)

        # STAGE 3: Grounded Zero-Context Cold-Scroll Critic (with Deterministic Score Gate)
        report(f"Stage 3/5 — Grounding & auditing {len(valid_candidates)} candidates via Zero-Context Critic...", 0.70)
        critic_reviews = self._run_critic_batch(valid_candidates, "critic_chunk", use_cache)

        approved_candidates: List[Dict[str, Any]] = []
        for c in valid_candidates:
            rev = critic_reviews.get(c["candidate_id"].upper(), {})
            c["critic"] = rev
            if rev.get("verified_hook_0_3s"):
                c["scout_hook_raw"] = c["hook_0_3s"]
                c["hook_0_3s"] = rev["verified_hook_0_3s"]
            if rev.get("verified_cliffhanger"):
                c["scout_cliffhanger_raw"] = c["cliffhanger"]
                c["cliffhanger"] = rev["verified_cliffhanger"]
            if rev.get("verified_title"):
                c["title"] = rev["verified_title"]
            if rev.get("verified_why_it_works"):
                c["why_it_works"] = rev["verified_why_it_works"]

            passed, total_score, reason = self._evaluate_critic_gate(rev)
            c["critic_total"] = total_score

            if not passed:
                rejected_log.append({
                    "candidate_id": c["candidate_id"],
                    "title": c["title"],
                    "stage": "cold_scroll_critic",
                    "reason": reason,
                    "scores": {
                        "hook_3s": rev.get("hook_3s"),
                        "standalone_clarity": rev.get("standalone_clarity"),
                        "cliffhanger_pull": rev.get("cliffhanger_pull"),
                    },
                    "segments_preview": [f"Ep{s['episode']} {s['start_ts']}-{s['end_ts']}" for s in c["segments"]],
                })
            else:
                approved_candidates.append(c)

        # Safety net ONLY if <5 passed strict gate (and always remove promoted IDs from rejected_log)
        if len(approved_candidates) < 5 and valid_candidates:
            scored = sorted(valid_candidates, key=lambda x: x.get("critic_total", 0), reverse=True)
            seen_ids = {c["candidate_id"] for c in approved_candidates}
            promoted_ids = set()
            for cand in scored:
                if cand["candidate_id"] not in seen_ids and cand.get("critic_total", 0) >= 9:
                    cand["promoted_from_strict_critic"] = True
                    approved_candidates.append(cand)
                    promoted_ids.add(cand["candidate_id"])
                    if len(approved_candidates) >= 7:
                        break
            rejected_log = [r for r in rejected_log if r.get("candidate_id") not in promoted_ids]

        # Verify stitch atoms & filter out atoms that heavily duplicate already-approved promos
        approved_segs = [s for c in approved_candidates for s in c["segments"]]
        verified_atoms = []
        for a_idx, atom in enumerate(all_stitch_atoms, start=1):
            res, err = self._resolve_segment(atom, snap_short=False, snap_long=False)
            if not res or not (8.0 <= res["duration_sec"] <= 26.0):
                continue
            overlaps_approved = any(self._seg_overlap_ratio(res, aseg) >= 0.60 for aseg in approved_segs)
            if overlaps_approved:
                continue
            verified_atoms.append({
                "atom_id": f"A{a_idx:02d}",
                "role": atom.get("role", "BEAT"),
                "episode": res["episode"],
                "start_beat_id": res["start_beat_id"],
                "end_beat_id": res["end_beat_id"],
                "duration_sec": res["duration_sec"],
                "summary": atom.get("summary", ""),
            })

        # STAGE 4A: Dedicated Flash-Forward Cross-Episode Stitcher
        pool: Dict[str, Dict[str, Any]] = {c["candidate_id"]: c for c in approved_candidates}
        if len(verified_atoms) >= 2:
            report(f"Stage 4/5 — Crafting flash-forward cross-episode stitches from {len(verified_atoms)} fresh atoms...", 0.82)
            stitcher_cache = self.cache_path / "stitcher_stage4a.json"
            st_res = self._cache_load(stitcher_cache, use_cache)
            if st_res is None:
                avoid_windows = [
                    f"Ep {s['episode']} ({s['start_ts']}-{s['end_ts']})"
                    for s in approved_segs
                ]
                st_res = self.llm.call_json(
                    STITCHER_SYSTEM_PROMPT,
                    json.dumps(
                        {
                            "series_summary": rolling_summary,
                            "avoid_windows": avoid_windows,
                            "stitch_atoms": verified_atoms[:35],
                        },
                        indent=2,
                    ),
                    model_override=self.strong_model,
                )
                stitcher_cache.write_text(json.dumps(st_res, indent=2), encoding="utf-8")

            valid_stitched: List[Dict[str, Any]] = []
            for st_idx, raw_st in enumerate(st_res.get("stitched_promos", []), start=1):
                st_id = f"ST{st_idx:02d}"
                status, payload = self._validate_promo(
                    raw_st,
                    st_id,
                    route_oversized=False,
                    snap_long=False,
                    require_multi_episode=True,
                )
                if status != "VALID":
                    payload["stage"] = "stitched_validator"
                    rejected_log.append(payload)
                    continue

                recycled_sec = sum(
                    seg["duration_sec"]
                    for seg in payload["segments"]
                    if any(self._seg_overlap_ratio(seg, aseg) >= 0.50 for aseg in approved_segs)
                )
                if recycled_sec / payload["total_duration_sec"] >= 0.50:
                    rejected_log.append({
                        "candidate_id": st_id,
                        "title": payload["title"],
                        "stage": "stitched_dedup",
                        "reason": "RECYCLES_APPROVED_CONTINUOUS_PROMO",
                    })
                    continue

                valid_stitched.append(payload)

            if valid_stitched:
                st_reviews = self._run_critic_batch(valid_stitched, "stitched_critic", use_cache)
                for st in valid_stitched:
                    rev = st_reviews.get(st["candidate_id"].upper(), {})
                    st["critic"] = rev
                    if rev.get("verified_hook_0_3s"):
                        st["hook_0_3s"] = rev["verified_hook_0_3s"]
                    if rev.get("verified_cliffhanger"):
                        st["cliffhanger"] = rev["verified_cliffhanger"]
                    if rev.get("verified_title"):
                        st["title"] = rev["verified_title"]
                    if rev.get("verified_why_it_works"):
                        st["why_it_works"] = rev["verified_why_it_works"]

                    passed, total_score, reason = self._evaluate_critic_gate(rev)
                    st["critic_total"] = total_score
                    if not passed:
                        rejected_log.append({
                            "candidate_id": st["candidate_id"],
                            "title": st["title"],
                            "stage": "stitched_critic",
                            "reason": reason,
                            "segments_preview": [f"Ep{s['episode']} {s['start_ts']}-{s['end_ts']}" for s in st["segments"]],
                        })
                    else:
                        pool[st["candidate_id"]] = st

        # STAGE 4B: Dedicated Global Ranker (Title-Anchored, Zero Cross-Wiring)
        report(f"Stage 5/5 — Globally ranking {len(pool)} verified promos & generating overlay hooks...", 0.92)
        ranker_cache = self.cache_path / "ranker_stage4b.json"
        rres = self._cache_load(ranker_cache, use_cache)
        if rres is None:
            compact_pool = [
                {
                    "id": c["candidate_id"],
                    "title": c["title"],
                    "format": c.get("promo_format", "Continuous"),
                    "trope": c["trope"],
                    "episodes": [s["episode"] for s in c["segments"]],
                    "duration_sec": c["total_duration_sec"],
                    "critic_total": c.get("critic_total", 0),
                    "critic_scores": {
                        "hook_3s": c.get("critic", {}).get("hook_3s"),
                        "standalone_clarity": c.get("critic", {}).get("standalone_clarity"),
                        "cliffhanger_pull": c.get("critic", {}).get("cliffhanger_pull"),
                    },
                    "verified_hook_0_3s": c["hook_0_3s"],
                    "verified_cliffhanger": c["cliffhanger"],
                    "why_it_works": c["why_it_works"],
                }
                for c in sorted(pool.values(), key=lambda x: x.get("critic_total", 0), reverse=True)
            ]
            rres = self.llm.call_json(
                RANKER_SYSTEM_PROMPT,
                json.dumps({"series_summary": rolling_summary, "verified_promos": compact_pool}, indent=2),
                model_override=self.strong_model,
            )
            ranker_cache.write_text(json.dumps(rres, indent=2), encoding="utf-8")

        final_ranked: List[Dict[str, Any]] = []
        used_ids = set()

        def _safe_rank(x: Any) -> int:
            if not isinstance(x, dict) or x.get("rank") is None:
                return 999
            try:
                return int(x["rank"])
            except (TypeError, ValueError):
                return 999

        for entry in sorted(rres.get("ranked_ids", []), key=_safe_rank):
            if not isinstance(entry, dict):
                continue
            pid = str(entry.get("id", "")).strip().upper()
            if pid in pool and pid not in used_ids:
                item = dict(pool[pid])
                tier = str(entry.get("posting_tier", "Wave 2 (Scale/Retarget)")).strip()
                if "Wave 1" in tier and item.get("critic_total", 0) < 12:
                    tier = "Wave 2 (Scale/Retarget)"
                item["rank"] = len(final_ranked) + 1
                item["posting_tier"] = tier
                reason = str(entry.get("ranking_reason", "")).strip()
                item["ranking_reason"] = reason if reason else item["why_it_works"]
                final_ranked.append(item)
                used_ids.add(pid)

        for pid, item in sorted(pool.items(), key=lambda kv: kv[1].get("critic_total", 0), reverse=True):
            if pid not in used_ids:
                copy_item = dict(item)
                copy_item["rank"] = len(final_ranked) + 1
                copy_item["posting_tier"] = "Backlog"
                copy_item["overlay_hook"] = ""
                copy_item["ranking_reason"] = copy_item["why_it_works"]
                final_ranked.append(copy_item)

        # STAGE 4C: Per-promo overlay writer. v4 wrote all overlays in ONE ranker call and
        # cross-wired them (#12 got another promo's "cathedral" text). Now each overlay is
        # written from its OWN clip only.
        report(f"Stage 5/5 — Writing overlays one promo at a time ({len(final_ranked)} promos)...", 0.95)
        for item in final_ranked:
            if item.get("posting_tier") == "Backlog":
                continue
            ov_cache = self.cache_path / f"overlay_{item['candidate_id']}.json"
            ores = self._cache_load(ov_cache, use_cache)
            if ores is None:
                clip_text = "\n--- SCENE CUT ---\n".join(s["transcript"] for s in item["segments"])
                try:
                    ores = self.llm.call_json(
                        OVERLAY_SYSTEM_PROMPT,
                        f"CLIP {item['candidate_id']} ({item['total_duration_sec']}s):\n{clip_text}",
                        model_override=self.strong_model,
                    )
                except RuntimeError as e:
                    ores = {"overlay_hook": "", "who_does_what_to_whom": f"OVERLAY_FAILED: {str(e)[:120]}"}
                ov_cache.write_text(json.dumps(ores, indent=2), encoding="utf-8")
            item["overlay_hook"] = str(ores.get("overlay_hook", "")).strip()
            item["overlay_who_does_what_to_whom"] = str(ores.get("who_does_what_to_whom", "")).strip()

        # Deterministic grounding audit: flags for a human, never auto-deletes.
        for item in final_ranked:
            item["review_flags"] = audit_promo(item, self.specific_words, self.character_names, self.ep_map)
        flagged_count = sum(1 for p in final_ranked if p["review_flags"])

        intra_count = sum(1 for p in final_ranked if p.get("promo_format") == "Intra-Episode Cut")
        cross_count = sum(1 for p in final_ranked if p.get("promo_format") == "Cross-Episode Stitched")

        # Honest usage numbers: a fully fresh run saves its own usage; a run that reused cache
        # reports what it spent now AND what the original fresh run cost.
        usage = self._usage()
        metrics_file = self.cache_path / "fresh_run_usage.json"
        if self.cache_hits == 0:
            metrics_file.write_text(json.dumps(usage, indent=2), encoding="utf-8")
        original_usage = (
            json.loads(metrics_file.read_text(encoding="utf-8"))
            if self.cache_hits > 0 and metrics_file.exists()
            else None
        )

        report("Done!", 1.0)
        return {
            "meta": {
                "episodes_processed": len(self.episodes),
                "total_beats_parsed": len(self.beat_index),
                "provider": self.llm.provider,
                "scout_model": self.llm.model,
                "critic_ranker_model": self.strong_model,
                "critic_provider": (self.critic_llm.provider if self.critic_llm else self.llm.provider),
                "critic_model": self.critic_model,
                **usage,
                "cache_hits": self.cache_hits,
                "fresh_run_usage": original_usage,
                "promos_flagged_for_review": flagged_count,
                "raw_candidates_found": len(all_raw_candidates),
                "oversized_sent_to_trimmer": len(oversized_queue),
                "recovered_by_trimmer": recovered_count,
                "intra_episode_cuts_approved": intra_count,
                "cross_episode_stitched_approved": cross_count,
                "final_ranked_count": len(final_ranked),
                "rejected_count": len(rejected_log),
            },
            "series_summary": rolling_summary,
            "ranked_promos": final_ranked,
            "rejected_audit": rejected_log,
        }


def export_outputs(result: Dict[str, Any], out_dir: str = "outputs") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / "promos_raw.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    csv_path = out / "ranked_promos.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Rank", "ID", "Tier", "Title", "Format", "Total_Sec",
            "Segments (Episode & Timestamps)", "Burned-In 0-3s Overlay",
            "Hook (0-3s)", "Cliffhanger", "Why It Works", "Ranking Reason",
            "Critic Scores (H/C/Cl)", "Critic Total (/15)",
            "Overlay Basis (who does what to whom)", "Auto Review Flags",
        ])
        for p in result["ranked_promos"]:
            segs_str = " + ".join(f"Ep {s['episode']} [{s['start_ts']} - {s['end_ts']}]" for s in p["segments"])
            cr = p.get("critic", {})
            c_score = f"{cr.get('hook_3s','-')}/{cr.get('standalone_clarity','-')}/{cr.get('cliffhanger_pull','-')}"
            fmt = p.get("promo_format") or ("Cross-Episode Stitched" if p.get("is_stitched") else "Continuous")
            writer.writerow([
                p["rank"], p["candidate_id"], p.get("posting_tier", ""), p["title"],
                fmt, p["total_duration_sec"], segs_str, p.get("overlay_hook", ""),
                p["hook_0_3s"], p["cliffhanger"], p["why_it_works"],
                p.get("ranking_reason", ""), c_score, p.get("critic_total", "-"),
                p.get("overlay_who_does_what_to_whom", ""), " | ".join(p.get("review_flags", [])),
            ])

    m = result["meta"]
    usage_line = f"- **API calls:** {m['llm_calls']} | **Tokens:** {m['prompt_tokens']:,} in / {m['completion_tokens']:,} out"
    if m.get("cache_hits"):
        fr = m.get("fresh_run_usage")
        usage_line += f" *(this run reused {m['cache_hits']} cached LLM results"
        usage_line += (
            f"; the original fresh run used {fr['llm_calls']} calls / "
            f"{fr['prompt_tokens'] + fr['completion_tokens']:,} tokens)*"
            if fr else "; fresh-run usage not recorded — rerun with --no-cache to measure)*"
        )
    critic_desc = f"{m.get('critic_provider', m['provider'])} / {m.get('critic_model', m['critic_ranker_model'])}"
    md = [
        "# Ranked Promo Report (v5)\n",
        f"- **Episodes processed:** {m['episodes_processed']} ({m['total_beats_parsed']:,} beats)",
        f"- **Scout model:** `{m['provider']} / {m['scout_model']}` | **Critic:** `{critic_desc}` | **Ranker & overlays:** `{m['critic_ranker_model']}`",
        usage_line,
        f"- **Auto review flags:** {m.get('promos_flagged_for_review', 0)} of {m['final_ranked_count']} promos need a human look (wrong place/person in text, or hook/cliffhanger text that doesn't match the actual first/last 3 seconds)",
        f"- **Funnel:** {m['raw_candidates_found']} scouted -> {m['oversized_sent_to_trimmer']} sent to Smart Trimmer ({m['recovered_by_trimmer']} recovered) -> **{m['final_ranked_count']} Approved & Ranked** ({m.get('intra_episode_cuts_approved', 0)} Intra-Episode Cuts, {m.get('cross_episode_stitched_approved', 0)} Cross-Episode Stitched) | **{m['rejected_count']} Rejected in Audit**\n",
        "---\n## Ranked Promos\n",
    ]
    for p in result["ranked_promos"]:
        cr = p.get("critic", {})
        c_tag = (
            f"Hook {cr.get('hook_3s','-')}/5 | Clarity {cr.get('standalone_clarity','-')}/5 "
            f"| Cliffhanger {cr.get('cliffhanger_pull','-')}/5 (**{p.get('critic_total','-')}/15**)"
        )
        fmt = p.get("promo_format") or ("Cross-Episode Stitched" if p.get("is_stitched") else "Continuous")
        trim_tag = f" *(compressed from {p['trimmed_from_sec']}s)*" if p.get("trimmed_from_sec") else ""
        seg_lines = "\n".join(
            f"  - Episode {s['episode']}: `{s['start_ts']} -> {s['end_ts']}` ({s['duration_sec']}s | beats `{s['start_beat_id']}..{s['end_beat_id']}`)"
            for s in p["segments"]
        )
        first_proof = " / ".join(p["segments"][0].get("opening_3s", [])[:2])
        last_proof = " / ".join(p["segments"][-1].get("closing_3s", [])[-2:])
        overlay_line = f"- **Suggested 0-3s Text Overlay:** \"{p['overlay_hook']}\"\n" if p.get("overlay_hook") else ""
        if p.get("overlay_who_does_what_to_whom"):
            overlay_line += f"  - *Overlay basis:* {p['overlay_who_does_what_to_whom']}\n"
        flags_line = "".join(f"- **NEEDS REVIEW:** {f}\n" for f in p.get("review_flags", []))
        trim_note_line = f"- **Compression Note:** {p['trim_note']}\n" if p.get("trim_note") else ""
        md.append(
            f"### #{p['rank']} [{p['posting_tier']}] {p['title']} ({p['total_duration_sec']}s){trim_tag}\n"
            f"- **ID:** `{p['candidate_id']}` | **Format:** {fmt} | **Trope:** {p['trope']}\n"
            f"- **Cold-Scroll Critic:** {c_tag}\n"
            f"{flags_line}"
            f"- **Segments:**\n{seg_lines}\n"
            f"{overlay_line}"
            f"{trim_note_line}"
            f"- **Verified 0-3s Hook:** {p['hook_0_3s']}\n"
            f"  - *Script proof (0-3s):* `{first_proof}`\n"
            f"- **Verified Cliffhanger:** {p['cliffhanger']}\n"
            f"  - *Script proof (last 3s):* `{last_proof}`\n"
            f"- **Why it works:** {p['why_it_works']}\n"
            f"- **Ranking rationale:** {p.get('ranking_reason', '')}\n"
        )

    md.append("\n---\n## Quality Audit: Rejected & Failed Picks\n")
    for r in result["rejected_audit"]:
        md.append(f"- **{r.get('candidate_id')} ({r.get('title', 'Untitled')})** — Rejected at `{r.get('stage')}`: {r.get('reason')}")

    (out / "promo_report.md").write_text("\n".join(md), encoding="utf-8")
    return out