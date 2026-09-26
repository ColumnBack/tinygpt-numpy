"""
TinyGPT inference

train.py 가 저장한 model.npz 를 불러와서,
한두 단어를 입력하면 이어지는 문장을 만들어 준다.

학습은 하지 않는다. 모델 구조와 단어장은 model.npz 에서 복원한다.

Usage:

    python generate.py
    python generate.py --temperature 1.0 --samples 5

학습은 train.py
"""

import argparse
import sys
from pathlib import Path

from tinygpt import (
    decode,
    load_model
)


# ====================================================================
# Interactive generation
#
# 사용자가 한두 단어를 입력하면
#
#   <BOS> + 입력 단어
#
# 를 prompt 로 해서 문장을 이어서 생성한다.
# ====================================================================

def show_corpus(
    sentences,
    token_to_id
):

    print()
    print("===== 학습 문장 =====")

    for i, sentence in enumerate(sentences, 1):

        print(
            f"{i:3d}. {sentence}"
        )

    # ---------------------------------------------------------------
    # 문장 첫 단어
    #
    # 입력은 <BOS> 뒤에 붙으므로 문장을 시작하는 단어로
    # 입력하면 가장 자연스럽다.
    # ---------------------------------------------------------------

    first_words = sorted(
        {
            sentence.lower().split()[0]
            for sentence in sentences
        }
    )

    words = sorted(
        token
        for token in token_to_id
        if not token.startswith("<")
    )

    print()
    print(
        "문장 첫 단어:",
        ", ".join(first_words)
    )

    print(
        f"전체 단어 ({len(words)}개):",
        ", ".join(words)
    )


def interactive(
    model,
    token_to_id,
    id_to_token,
    sentences,
    n_samples=3,
    temperature=0.7
):

    known_sentences = {
        sentence.lower()
        for sentence in sentences
    }

    print()
    print(
        "한두 단어를 입력하면 문장을 만들어 줍니다. "
        "(예: the cat / my / we)  종료: q 또는 빈 줄"
    )

    while True:

        try:

            # 파이프 입력 시 붙을 수 있는 BOM 제거
            text = input("\n> ").replace("﻿", "").strip().lower()

        except (EOFError, KeyboardInterrupt):

            print()
            break

        if text in ("", "q", "quit", "exit"):
            break

        words = text.split()

        # ------------------------------------------------------------
        # 학습에 없는 단어는 <UNK> 로 바뀌는데, <UNK> 는 학습된 적이
        # 없어서 의미 있는 문장이 나오지 않는다. 그래서 먼저 알려준다.
        # ------------------------------------------------------------

        unknown = [
            w
            for w in words
            if w not in token_to_id
            or w.startswith("<")
        ]

        if unknown:

            print(
                "  학습 문장에 없는 단어입니다:",
                ", ".join(unknown)
            )

            continue

        prompt_ids = (
            [token_to_id["<BOS>"]]
            +
            [token_to_id[w] for w in words]
        )

        results = []

        for _ in range(n_samples):

            generated_ids = model.generate(
                prompt_ids,
                id_to_token,
                max_new_tokens=model.max_context,
                temperature=temperature
            )

            sentence = decode(
                generated_ids,
                id_to_token
            )

            if sentence not in results:
                results.append(sentence)

        for sentence in results:

            tag = (
                "학습 문장"
                if sentence in known_sentences
                else "새 조합"
            )

            print(
                f"  - {sentence}   [{tag}]"
            )


# ====================================================================
# Main
# ====================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="TinyGPT inference"
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).with_name("model.npz"),
        help="불러올 모델 파일 (default: model.npz)"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="낮을수록 학습 문장에 가깝게, 높을수록 다양하게"
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="입력마다 만들 문장 수"
    )

    args = parser.parse_args()

    if not args.model.exists():

        print(
            f"{args.model} 가 없습니다. "
            "먼저 python train.py 로 학습하세요."
        )

        sys.exit(1)

    (
        model,
        token_to_id,
        id_to_token,
        sentences,
        epoch
    ) = load_model(
        args.model
    )

    parameter_count = sum(
        parameter.size
        for parameter in model.p.values()
    )

    print(
        f"모델: {args.model.name} | {epoch} epoch 학습 | "
        f"vocab {len(token_to_id)} | parameters {parameter_count}"
    )

    show_corpus(
        sentences,
        token_to_id
    )

    interactive(
        model,
        token_to_id,
        id_to_token,
        sentences,
        n_samples=args.samples,
        temperature=args.temperature
    )
