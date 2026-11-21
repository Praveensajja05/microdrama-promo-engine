from pathlib import Path

from parser import format_timestamp, load_episodes_from_path, parse_timestamp

FAKE_SHOW = Path(__file__).resolve().parent.parent / "samples" / "fake_show_boardroom"


def test_timestamps_round_trip():
    assert parse_timestamp("02:18.00") == 138.0
    assert parse_timestamp("(00:04.80)") == 4.8
    assert format_timestamp(138.0) == "02:18.00"


def test_fake_show_parses_in_primary_mode():
    episodes = load_episodes_from_path(str(FAKE_SHOW))
    assert [e.episode_num for e in episodes] == [1, 2, 3]
    for ep in episodes:
        assert ep.parser_mode == "kuku_primary"
        assert len(ep.beats) >= 20
        assert 60 <= ep.duration_sec <= 75
        # beats are sorted and IDs are unique
        starts = [b.start_sec for b in ep.beats]
        assert starts == sorted(starts)
        assert len({b.beat_id for b in ep.beats}) == len(ep.beats)


def test_speaker_labels_with_parentheticals_are_dialogue():
    ep1 = load_episodes_from_path(str(FAKE_SHOW))[0]
    dialogue = [b for b in ep1.beats if b.kind == "DIALOGUE"]
    assert any(b.speaker.startswith("OLD MAN (V.O.)") for b in dialogue)
    assert all(b.kind != "DIALOGUE" for b in ep1.beats if b.speaker in {"CAMERA", "ACTION", "SOUND FX"})
