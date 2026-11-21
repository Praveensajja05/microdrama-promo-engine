import sys
from parser import load_episodes_from_path

path = sys.argv[1] if len(sys.argv) > 1 else "data/series.zip"
episodes = load_episodes_from_path(path)
print(f"Total episodes loaded: {len(episodes)}")

bad = []
seen_nums = set()
dups = []

for ep in episodes:
    if ep.episode_num in seen_nums:
        dups.append(ep.episode_num)
    seen_nums.add(ep.episode_num)

    flag = ""
    if ep.parser_mode != "kuku_primary" or len(ep.beats) < 20 or ep.duration_sec < 45:
        flag = "  <-- CHECK THIS EPISODE"
        bad.append(ep.episode_num)
    print(f"Ep {ep.episode_num:02d} | beats={len(ep.beats):3d} | dur={ep.duration_sec:6.2f}s | mode={ep.parser_mode}{flag}")

print(f"\nTotal beats: {sum(len(e.beats) for e in episodes):,}")
print(f"Duplicate episode numbers: {dups if dups else 'NONE'}")
print(f"Suspicious episodes: {bad if bad else 'NONE — parser clean across all episodes!'}")