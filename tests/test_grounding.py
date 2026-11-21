from pathlib import Path

from grounding import audit_promo, borrowed_words, character_words, content_words, place_and_people_words
from parser import load_episodes_from_path

FAKE_SHOW = Path(__file__).resolve().parent.parent / "samples" / "fake_show_boardroom"


def _promo(segment_text, opening, closing, overlay="", hook="", cliff=""):
    return {
        "segments": [{"transcript": segment_text, "opening_3s": opening, "closing_3s": closing}],
        "overlay_hook": overlay,
        "hook_0_3s": hook,
        "cliffhanger": cliff,
    }


def test_content_words_strip_beat_prefixes_and_stopwords():
    words = content_words('[E01_B012 | 11.3s-13.1s] CAM: Close-up of Mia crying in the boardroom')
    assert "boardroom" in words and "crying" in words
    assert "close" not in words and "e01" not in words


def test_place_from_another_episode_is_flagged():
    # The v4 bug: an overlay naming a place that is not in its own clip.
    eps = load_episodes_from_path(str(FAKE_SHOW))
    specific = place_and_people_words(eps)
    ep1_text = eps[0].to_prompt_text()  # includes scene headings, like the pipeline
    assert "penthouse" in specific                      # Episode 2's location
    assert "penthouse" in borrowed_words("Heiress humiliated at the penthouse party", ep1_text, specific)
    assert borrowed_words("Assistant humiliated in the boardroom", ep1_text, specific) == []


def test_generic_paraphrase_words_are_not_flagged():
    eps = load_episodes_from_path(str(FAKE_SHOW))
    specific = place_and_people_words(eps)
    ep1_text = eps[0].to_prompt_text()  # includes scene headings, like the pipeline
    # 'brutally', 'declares' never appear in the clip but are paraphrase, not facts
    assert borrowed_words("Boss brutally declares her fired", ep1_text, specific) == []


def test_cliffhanger_that_describes_an_earlier_beat_is_flagged():
    eps = load_episodes_from_path(str(FAKE_SHOW))
    specific, names = place_and_people_words(eps), character_words(eps)
    closing = ['[E01_B030 | 62.5s-66.5s] MIA (Deadly Calm): "Someone just showed me what this company has become."']
    good = _promo("x", [], closing, cliff="Mia vows the company has become something she will fix")
    bad = _promo("x", [], closing, cliff="Chloe flashes the diamond engagement ring")
    assert not any("CLIFFHANGER text does not match" in f for f in audit_promo(good, specific, names))
    assert any("CLIFFHANGER text does not match" in f for f in audit_promo(bad, specific, names))


def test_reversed_meaning_is_a_known_blind_spot():
    # Documented limitation: same words, flipped meaning -> NOT caught. A human must review.
    eps = load_episodes_from_path(str(FAKE_SHOW))
    specific, names = place_and_people_words(eps), character_words(eps)
    clip = 'MIA: "Lock down the tower. Nobody leaves. Especially Victor."'
    p = _promo(clip, [clip], [clip], overlay="Victor locks Mia in the tower")
    assert audit_promo(p, specific, names) == []
