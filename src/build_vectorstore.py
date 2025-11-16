"""CLI helper that builds the vectorstore JSON from data/files."""

from __future__ import annotations

import argparse
from pathlib import Path

from .vector_store import DEFAULT_VECTORSTORE_PATH, create_vectorstore


def main() -> None:
    """Entry point for building/updating vectorstore.json."""
    parser = argparse.ArgumentParser(description="LangGraph 벡터스토어 생성기")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_VECTORSTORE_PATH,
        help="생성된 벡터스토어 JSON 경로",
    )
    args = parser.parse_args()

    output_path = create_vectorstore(args.output)
    print(f"벡터스토어 생성 완료: {output_path}")


if __name__ == "__main__":
    main()
