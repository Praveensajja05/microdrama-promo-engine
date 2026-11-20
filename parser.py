import io
import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

MOJIBAKE_MAP = {
    "â€“": "–",
    "â€”": "—",
    "â€™": "'",
    "â€œ": '"',
    "â€\x9d": '"',
    "Ã©": "é",
    "Ã¨": "è",
    "Â": "",
}


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    for bad, good in MOJIBAKE_MAP.items():
        text = text.replace(bad, good)
    return text


def parse_timestamp(ts: str) -> float:
    """Converts MM:SS.cc or HH:MM:SS.cc into float seconds."""
    ts = ts.strip().strip("()[]")
    parts = ts.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    raise ValueError(f"Unrecognized timestamp: {ts}")


def format_timestamp(seconds: float) -> str:
    """Formats float seconds as MM:SS.cc."""
    seconds = max(0.0, seconds)
    mins = int(seconds // 60)
    secs = seconds - mins * 60
    return f"{mins:02d}:{secs:05.2f}"


@dataclass
class Beat:
    beat_id: str
    episode: int
    scene: str
    kind: str        # CAMERA | ACTION | SOUND_FX | DIALOGUE
    speaker: str     # Raw speaker/label
    start_sec: float
    end_sec: float
    text: str

    def to_compact_line(self) -> str:
        ts = f"{self.start_sec:.1f}s-{self.end_sec:.1f}s"
        if self.kind == "CAMERA":
            tag = "CAM"
        elif self.kind == "ACTION":
            tag = "ACT"
        elif self.kind == "SOUND_FX":
            tag = "SFX"
        else:
            tag = self.speaker
        clean_body = re.sub(r"\s+", " ", self.text).strip()
        return f"[{self.beat_id} | {ts}] {tag}: {clean_body}"


@dataclass
class Episode:
    episode_num: int
    filename: str
    logline: str
    continuity: str
    duration_sec: float
    beats: List[Beat]
    parser_mode: str  # "kuku_primary" or "generic_fallback"

    def to_prompt_text(self) -> str:
        header = f"=== EPISODE {self.episode_num} (Duration: {format_timestamp(self.duration_sec)}) ==="
        if self.logline:
            header += f"\nLOGLINE: {self.logline}"
        lines = [header]
        current_scene = None
        for b in self.beats:
            if b.scene and b.scene != current_scene:
                lines.append(f"-- {b.scene} --")
                current_scene = b.scene
            lines.append(b.to_compact_line())
        return "\n".join(lines)


def _extract_episode_num(text: str, filename: str, fallback_idx: int) -> int:
    m = re.search(r"EPISODE\s+NUMBER\s*:\s*(?:Episode\s*)?(\d+)", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"(?:episode|ep)[-_ ]*0*(\d+)", filename, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return fallback_idx


def _classify_label(label: str) -> Tuple[str, str]:
    up = label.strip().upper()
    if up == "CAMERA":
        return "CAMERA", "CAMERA"
    if up == "ACTION":
        return "ACTION", "ACTION"
    if up in {"SOUND FX", "SFX", "SOUND", "AUDIO"}:
        return "SOUND_FX", "SOUND FX"
    return "DIALOGUE", label.strip()


def _parse_kuku_primary(body: str, ep_num: int) -> List[Beat]:
    """Parses Kuku microdrama screenplays with case-insensitive scene & beat matching."""
    scene_pattern = re.compile(r"^SCENE\s+\d+.*$", re.IGNORECASE | re.MULTILINE)
    heading_pattern = re.compile(r"^Scene\s+heading\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
    block_pattern = re.compile(
        r"\*\*(.+?):\*\*\s*\n\s*Start\s*:\s*\(?([0-9:.]+)\)?\s*\|\s*End\s*:\s*\(?([0-9:.]+)\)?\s*\n(.*?)(?=\n\*\*.+?:\*\*|\nSCENE\s+\d+|\n---\s*SCRIPT\s+END|\Z)",
        re.IGNORECASE | re.DOTALL,
    )

    scene_spans: List[Tuple[int, str]] = []
    for sm in scene_pattern.finditer(body):
        chunk_after = body[sm.start() : sm.start() + 300]
        hm = heading_pattern.search(chunk_after)
        label = hm.group(1).strip() if hm else sm.group(0).strip()
        scene_spans.append((sm.start(), label))

    def scene_at(pos: int) -> str:
        current = "SCENE 01"
        for s_pos, s_label in scene_spans:
            if s_pos <= pos:
                current = s_label
            else:
                break
        return current

    raw_beats: List[Tuple[float, float, str, str, str, str]] = []
    for m in block_pattern.finditer(body):
        label = m.group(1).strip()
        try:
            start_s = parse_timestamp(m.group(2))
            end_s = parse_timestamp(m.group(3))
        except ValueError:
            continue
        if end_s < start_s:
            start_s, end_s = end_s, start_s
        content = re.sub(r"\s+", " ", m.group(4)).strip()
        if not content:
            continue
        kind, speaker = _classify_label(label)
        raw_beats.append((start_s, end_s, scene_at(m.start()), kind, speaker, content))

    raw_beats.sort(key=lambda x: (x[0], x[1]))

    beats: List[Beat] = []
    for idx, (s, e, scene, kind, speaker, content) in enumerate(raw_beats, start=1):
        beats.append(
            Beat(
                beat_id=f"E{ep_num:02d}_B{idx:03d}",
                episode=ep_num,
                scene=scene,
                kind=kind,
                speaker=speaker,
                start_sec=round(s, 2),
                end_sec=round(e, 2),
                text=content,
            )
        )
    return beats


def _parse_generic_fallback(text: str, ep_num: int) -> List[Beat]:
    """Fallback parser for unseen interview screenplay formats (e.g., [00:12-00:16] or 00:12 Speaker: Line)."""
    range_pat = re.compile(r"[\[(]?(\d{1,2}:\d{2}(?:\.\d+)?)\s*[-–—to]+\s*(\d{1,2}:\d{2}(?:\.\d+)?)[\])]?\s*(.+)")
    single_pat = re.compile(r"[\[(]?(\d{1,2}:\d{2}(?:\.\d+)?)[\])]?\s*(.+)")

    rows: List[Tuple[float, float, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rm = range_pat.match(line)
        if rm:
            rows.append((parse_timestamp(rm.group(1)), parse_timestamp(rm.group(2)), rm.group(3).strip()))
            continue
        sm = single_pat.match(line)
        if sm:
            rows.append((parse_timestamp(sm.group(1)), -1.0, sm.group(2).strip()))

    beats: List[Beat] = []
    rows.sort(key=lambda x: x[0])
    for i, (s, e, content) in enumerate(rows):
        if e < 0:
            e = rows[i + 1][0] if i + 1 < len(rows) else s + 3.0
        if e <= s:
            e = s + 2.0
        speaker = "NARRATION"
        kind = "DIALOGUE"
        if ":" in content and len(content.split(":", 1)[0]) < 35:
            sp, rest = content.split(":", 1)
            kind, speaker = _classify_label(sp)
            content = rest.strip()
        beats.append(
            Beat(
                beat_id=f"E{ep_num:02d}_B{i+1:03d}",
                episode=ep_num,
                scene="SCENE 01",
                kind=kind,
                speaker=speaker,
                start_sec=round(s, 2),
                end_sec=round(e, 2),
                text=content,
            )
        )
    return beats


def parse_episode_text(raw_text: str, filename: str, fallback_idx: int = 1) -> Episode:
    text = clean_text(raw_text)
    ep_num = _extract_episode_num(text, filename, fallback_idx)

    log_m = re.search(r"EPISODE\s+LOGLINE\s*:\s*(.+)", text, re.IGNORECASE)
    cont_m = re.search(r"CONTINUITY\s+NOTES\s*:\s*(.+)", text, re.IGNORECASE)
    logline = log_m.group(1).strip() if log_m else ""
    continuity = cont_m.group(1).strip() if cont_m else ""

    start_m = re.search(r"---\s*SCRIPT\s+START\s*---", text, re.IGNORECASE)
    end_m = re.search(r"---\s*SCRIPT\s+END\s*---", text, re.IGNORECASE)
    body = text[start_m.end() : end_m.start() if end_m else len(text)] if start_m else text

    beats = _parse_kuku_primary(body, ep_num)
    mode = "kuku_primary"
    if len(beats) < 3:
        beats = _parse_generic_fallback(body, ep_num)
        mode = "generic_fallback"

    duration = max((b.end_sec for b in beats), default=0.0)
    return Episode(
        episode_num=ep_num,
        filename=filename,
        logline=logline,
        continuity=continuity,
        duration_sec=round(duration, 2),
        beats=beats,
        parser_mode=mode,
    )


def load_episodes_from_path(path: str) -> List[Episode]:
    """Loads episodes from a folder of .txt files, a .zip archive, or a merged === FILE: === .txt file."""
    p = Path(path)
    raw_files: List[Tuple[str, str]] = []

    if p.is_dir():
        for f in sorted(p.rglob("*.txt")):
            if f.name.lower() == "all_episodes.txt":
                continue
            raw_files.append((f.name, f.read_text(encoding="utf-8-sig", errors="replace")))
    elif p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p, "r") as zf:
            for name in sorted(zf.namelist()):
                if name.lower().endswith(".txt") and not name.startswith("__MACOSX"):
                    raw_files.append((os.path.basename(name), zf.read(name).decode("utf-8-sig", errors="replace")))
    elif p.is_file():
        content = p.read_text(encoding="utf-8-sig", errors="replace")
        chunks = re.split(r"===\s*FILE:\s*(.+?)\s*===", content)
        if len(chunks) > 2:
            for i in range(1, len(chunks), 2):
                raw_files.append((chunks[i].strip(), chunks[i + 1]))
        else:
            raw_files.append((p.name, content))
    else:
        raise FileNotFoundError(f"Input path not found: {p.resolve()}")

    episodes = [parse_episode_text(txt, fname, idx + 1) for idx, (fname, txt) in enumerate(raw_files)]
    episodes.sort(key=lambda ep: ep.episode_num)
    return episodes


def load_episodes_from_bytes(uploaded_files: List[Tuple[str, bytes]]) -> List[Episode]:
    """Loads episodes directly from Streamlit uploaded files (.zip or multiple .txt)."""
    raw_files: List[Tuple[str, str]] = []
    for fname, data in uploaded_files:
        if fname.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
                for name in sorted(zf.namelist()):
                    if name.lower().endswith(".txt") and not name.startswith("__MACOSX"):
                        raw_files.append((os.path.basename(name), zf.read(name).decode("utf-8-sig", errors="replace")))
        elif fname.lower().endswith(".txt"):
            text = data.decode("utf-8-sig", errors="replace")
            chunks = re.split(r"===\s*FILE:\s*(.+?)\s*===", text)
            if len(chunks) > 2:
                for i in range(1, len(chunks), 2):
                    raw_files.append((chunks[i].strip(), chunks[i + 1]))
            else:
                raw_files.append((fname, text))

    episodes = [parse_episode_text(txt, fname, idx + 1) for idx, (fname, txt) in enumerate(raw_files)]
    episodes.sort(key=lambda ep: ep.episode_num)
    return episodes