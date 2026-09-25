import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent))

from src.pipeline import EntityResolutionPipeline


def main():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026 - Entity Resolution Pipeline")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config YAML")
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold override")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory override")
    args = parser.parse_args()

    pipeline = EntityResolutionPipeline(config_path=args.config)
    results = pipeline.run(threshold=args.threshold, output_dir=args.output_dir)

    print("\n" + "=" * 50)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 50)
    for k, v in results.items():
        print(f"  {k}: {v}")
    print("=" * 50)


if __name__ == "__main__":
    main()
