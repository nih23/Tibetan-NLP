#!/usr/bin/env python3
"""Unified PechaBridge CLI entrypoint for texture augmentation workflow."""

from __future__ import annotations

import argparse
import logging

from tibetan_utils.arg_utils import (
    create_prepare_texture_lora_dataset_parser,
    create_texture_augment_parser,
    create_train_texture_lora_parser,
)

LOGGER = logging.getLogger("pechabridge_cli")


def _build_root_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PechaBridge command line interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parent = create_prepare_texture_lora_dataset_parser(add_help=False)
    prepare_parser = subparsers.add_parser(
        "prepare-texture-lora-dataset",
        parents=[prepare_parent],
        help="Prepare real-page texture crops + JSONL metadata for LoRA training",
        description=prepare_parent.description,
    )
    prepare_parser.set_defaults(handler=_run_prepare_texture_lora_dataset)

    train_parent = create_train_texture_lora_parser(add_help=False)
    train_parser = subparsers.add_parser(
        "train-texture-lora",
        parents=[train_parent],
        help="Train SDXL texture LoRA adapters using accelerate",
        description=train_parent.description,
    )
    train_parser.set_defaults(handler=_run_train_texture_lora)

    augment_parent = create_texture_augment_parser(add_help=False)
    augment_parser = subparsers.add_parser(
        "texture-augment",
        parents=[augment_parent],
        help="Apply SDXL + ControlNet Canny texture augmentation",
        description=augment_parent.description,
    )
    augment_parser.set_defaults(handler=_run_texture_augment)

    return parser


def _run_prepare_texture_lora_dataset(args: argparse.Namespace) -> int:
    from scripts.prepare_texture_lora_dataset import run

    run(args)
    return 0


def _run_train_texture_lora(args: argparse.Namespace) -> int:
    from scripts.train_texture_lora_sdxl import run

    run(args)
    return 0


def _run_texture_augment(args: argparse.Namespace) -> int:
    from scripts.texture_augment import run

    run(args)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    parser = _build_root_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.error("No subcommand selected")
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
