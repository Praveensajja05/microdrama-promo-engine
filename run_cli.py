import argparse
import os
from dotenv import load_dotenv
from llm_client import PROVIDER_PRESETS, SwappableLLMClient
from parser import load_episodes_from_path
from pipeline import PromoPipeline, export_outputs

load_dotenv()


def main():
    ap = argparse.ArgumentParser(description="Microdrama Promo Engine (Nemotron Primary + Gemini/OpenAI Backups)")
    ap.add_argument("--input", default="data/series.zip", help="Path to .zip, episodes folder, or merged .txt")
    ap.add_argument("--provider", default=os.getenv("LLM_PROVIDER", "nvidia"), choices=list(PROVIDER_PRESETS.keys()))
    ap.add_argument("--model", default=os.getenv("LLM_MODEL"), help="Override primary Scout model")
    ap.add_argument("--strong-model", default=os.getenv("LLM_STRONG_MODEL"), help="Override primary Critic/Ranker model")
    ap.add_argument("--batch-size", type=int, default=5)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    episodes = load_episodes_from_path(args.input)
    print(f"Loaded {len(episodes)} episodes ({sum(len(e.beats) for e in episodes):,} beats).")

    preset = PROVIDER_PRESETS[args.provider]
    scout_m = args.model or preset["default_model"]
    strong_m = args.strong_model or preset["strong_model"]

    llm = SwappableLLMClient(provider=args.provider, model=scout_m, strong_model=strong_m)
    print("Active failover chain:", " -> ".join(f"{e['provider']} ({e['model']})" for e in llm.chain))

    pipe = PromoPipeline(episodes=episodes, llm=llm, strong_model=strong_m, batch_size=args.batch_size)
    result = pipe.run(progress_cb=lambda msg, frac: print(f"[{int(frac*100):3d}%] {msg}"), use_cache=not args.no_cache)
    result["meta"]["failover_events"] = llm.failover_events

    out_path = export_outputs(result, args.out)
    print(f"\nDone! Exported outputs to: {out_path.resolve()}")


if __name__ == "__main__":
    main()