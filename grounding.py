"""Deterministic grounding checks for LLM-written promo text.

No LLM calls here. These checks target two failure modes found in the v4 run:

1. WRONG PLACE / PERSON: the overlay, hook or cliffhanger names a place or a person
   that does not occur anywhere in its own clip. Example from v4: promo #12 (Episode 33)
   got the overlay "...in cathedral nave", but Episode 33 never mentions a cathedral.
   That text was cross-wired from another promo.

2. DESCRIBES THE WRONG BEATS: the "cliffhanger" text shares no content words with the
   clip's actual last 3 seconds (or the "hook" text with its first 3 seconds). Example
   from v4: promo #8's cliffhanger described a blood close-up ~10s before the cut.

What these checks CANNOT catch: a reversed meaning that uses words from the clip
(v4 promo #1: "Husband declares lethal intent on bride", when the husband is the one
protecting his wife). Only a human reviewer catches that, which is why every promo
still needs a human sign-off before posting.
"""

import re
from typing import Dict, Iterable, List, Optional, Set

STOPWORDS = {
    "about", "above", "after", "again", "against", "also", "along", "already", "always",
    "another", "around", "away", "back", "because", "been", "before", "behind", "being",
    "below", "between", "both", "came", "cannot", "come", "could", "does", "doing",
    "down", "during", "each", "even", "every", "from", "further", "gets", "give", "goes",
    "going", "have", "having", "here", "herself", "himself", "into", "itself", "just",
    "keep", "know", "last", "like", "look", "looks", "made", "make", "many", "more", "most",
    "much", "must", "never", "next", "only", "onto", "other", "over", "same", "says",
    "should", "some", "still", "such", "take", "than", "that", "their", "them", "themselves",
    "then", "there", "these", "they", "thing", "this", "those", "through", "toward", "towards",
    "under", "until", "upon", "very", "want", "were", "what", "when", "where", "which",
    "while", "will", "with", "within", "without", "would", "your", "yours", "yourself",
    # screenplay / camera vocabulary that is not story content
    "shot", "close", "closeup", "medium", "wide", "angle", "frame", "framing", "camera",
    "insert", "tracking", "push", "pull", "profile", "extreme", "tight", "slow", "slowly",
    "cut", "cuts", "black", "scene", "beat", "beats", "segment", "clip", "seconds",
    "first", "final", "opening", "ending", "moment", "suddenly", "immediately",
}

# Words in scene headings that are not locations
HEADING_NOISE = {
    "continuous", "later", "morning", "evening", "night", "dusk", "dawn", "same", "time",
    "moments", "flashback", "intercut", "scene",
}


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def content_words(text: str) -> Set[str]:
    """Lower-case, stemmed content words of length >= 4, minus stopwords."""
    text = re.sub(r"\[[^\]]*\]", " ", text or "")  # drop "[E01_B012 | 11.3s-13.1s]" prefixes
    words = re.findall(r"[a-z]+", text.lower())
    return {_stem(w) for w in words if len(w) >= 4 and w not in STOPWORDS}


# Words in speaker labels that are not names, e.g. "VOICE (O.S.)", "ON SCREEN"
SPEAKER_NOISE = {"voice", "screen", "offscreen", "narrator", "narration", "cont", "continued", "overlap"}


def character_words(episodes: Iterable) -> Set[str]:
    """Words from dialogue speaker labels (names / roles), parentheticals removed."""
    names: Set[str] = set()
    for ep in episodes:
        for b in ep.beats:
            if b.kind == "DIALOGUE":
                names |= content_words(re.sub(r"\(.*?\)", " ", b.speaker))
    return names - SPEAKER_NOISE


def place_and_people_words(episodes: Iterable) -> Set[str]:
    """Words that name WHERE (scene-heading locations) or WHO (speaker labels).

    These make cross-wired text detectable: if an overlay names a place or a person that
    is not in its own clip, it was almost certainly copied from another promo. Generic
    verbs/adjectives ('declares', 'coldly') are excluded on purpose: they are normal
    paraphrase, and checking them flagged 16 of 20 v4 promos, almost all false alarms.
    """
    episodes = list(episodes)
    words: Set[str] = set()
    for ep in episodes:
        for b in ep.beats:
            words |= content_words(b.scene) - HEADING_NOISE
    return words | character_words(episodes)


def borrowed_words(text: str, clip_text: str, specific_words: Set[str]) -> List[str]:
    """Place/person words used in `text` that do not occur anywhere in this clip."""
    clip = content_words(clip_text)
    return sorted(w for w in content_words(text) if w in specific_words and w not in clip)


def shares_content_with(description: str, beat_lines: List[str], ignore: Set[str]) -> bool:
    """True if the description shares at least one non-name content word with the beats."""
    desc = content_words(description) - ignore
    beats = content_words(" ".join(beat_lines)) - ignore
    return bool(desc & beats)


def segment_scene_text(seg: Dict, ep_map: Optional[Dict]) -> str:
    """Scene headings of the beats inside a segment. Clip transcripts don't carry the
    heading, so without this a correct overlay naming the clip's own location
    (e.g. 'penthouse' for a penthouse scene) would be flagged."""
    if not ep_map or seg.get("episode") not in ep_map:
        return ""
    ep = ep_map[seg["episode"]]
    s, e = seg.get("start_sec", 0.0), seg.get("end_sec", 0.0)
    return " ".join(sorted({b.scene for b in ep.beats if s <= b.start_sec <= e}))


def audit_promo(promo: Dict, specific_words: Set[str], names: Set[str], ep_map: Optional[Dict] = None) -> List[str]:
    """Returns human-readable review flags for one promo (empty list = no flags).

    ep_map (episode_num -> Episode) lets the check include each segment's scene headings.
    """
    segs = promo.get("segments", [])
    if not segs:
        return ["NO_SEGMENTS"]
    clip_text = "\n".join(s.get("transcript", "") + "\n" + segment_scene_text(s, ep_map) for s in segs)
    flags: List[str] = []

    for field, label in (("overlay_hook", "OVERLAY"), ("hook_0_3s", "HOOK"), ("cliffhanger", "CLIFFHANGER")):
        bw = borrowed_words(promo.get(field, ""), clip_text, specific_words)
        if bw:
            flags.append(f"{label} names a place/person not in this clip: {', '.join(bw)}")

    opening = segs[0].get("opening_3s", [])
    closing = segs[-1].get("closing_3s", [])
    if promo.get("hook_0_3s") and opening and not shares_content_with(promo["hook_0_3s"], opening, names):
        flags.append("HOOK text does not match the clip's first 3 seconds")
    if promo.get("cliffhanger") and closing and not shares_content_with(promo["cliffhanger"], closing, names):
        flags.append("CLIFFHANGER text does not match the clip's last 3 seconds")
    return flags
