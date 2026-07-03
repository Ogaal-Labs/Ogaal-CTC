#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or update a Hugging Face model repo from a prepared local folder.")
    parser.add_argument("--folder", type=Path, required=True, help="Built model repo folder, typically artifacts/hf_model_repo.")
    parser.add_argument("--repo-id", required=True, help="Target Hugging Face repo id, for example user/model-name.")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--token", default=None, help="Optional HF token. If omitted, local cached auth is used.")
    parser.add_argument("--commit-message", default="Upload Ogaal CTC model release")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.folder.exists():
        raise FileNotFoundError(args.folder)

    api = HfApi()
    repo_url = api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
        token=args.token,
    )
    commit_info = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=args.folder,
        commit_message=args.commit_message,
        token=args.token,
    )
    print(
        json.dumps(
            {
                "repo_url": str(repo_url),
                "commit_url": getattr(commit_info, "commit_url", None),
                "oid": getattr(commit_info, "oid", None),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
