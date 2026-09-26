"""
TinyGPT training

학습 문장으로 TinyGPT 를 학습하고 model.npz 에 저장한다.

- 매 epoch 끝마다 model.npz 에 저장 (컴퓨터를 껐다 켜도 남음)
- Ctrl+C 로 중단하면 진행 중인 step 까지 반영해서 저장
- 다시 실행하면 저장된 epoch 다음부터 이어서 학습
- 학습 문장이나 모델 구조가 바뀌면 처음부터 다시 학습

Usage:

    python train.py
    python train.py --epochs 500     # 저장된 모델을 500 epoch 까지 이어서 학습
    python train.py --retrain        # 처음부터 다시 학습

문장 생성(추론)은 generate.py
"""

import argparse
import signal
from pathlib import Path

import numpy as np

from tinygpt import (
    TinyGPT,
    Adam,
    build_dataset,
    save_state,
    load_state
)


# ====================================================================
# Training
# ====================================================================

def train(
    model,
    data,
    epochs=1000,
    lr=2e-3,
    print_every=100,
    checkpoint_dir=None,
    checkpoint_every=1,
    optimizer=None,
    start_epoch=1,
    state_path=None,
    sentences=None
):

    # ---------------------------------------------------------------
    # state_path 가 주어지면 매 epoch 끝마다 학습 상태를 저장한다.
    #
    # Ctrl+C 를 누르면 진행 중인 step 까지만 마치고 저장한 뒤 멈춘다.
    # (한 번 더 누르면 강제 종료. 이때도 직전 epoch 상태는 남아 있다.)
    #
    # 학습을 끝까지 마치면 True, 중단되면 False 를 돌려준다.
    # ---------------------------------------------------------------

    # Save the model parameters every checkpoint_every epochs when a
    # checkpoint directory is provided.  Each .npz file contains every
    # array in model.p, using the parameter names as keys.
    if checkpoint_dir is not None:

        checkpoint_dir = Path(
            checkpoint_dir
        )

        checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    if optimizer is None:

        optimizer = Adam(
            model.p,
            lr=lr
        )

    # ---------------------------------------------------------------
    # Ctrl+C handler
    # ---------------------------------------------------------------

    stop = {"requested": False}

    def request_stop(signum, frame):

        if stop["requested"]:
            raise KeyboardInterrupt

        stop["requested"] = True

        print(
            "\n중단 요청: 현재 step 을 마치고 저장합니다. "
            "(강제 종료: Ctrl+C 한 번 더)"
        )

    previous_handler = signal.signal(
        signal.SIGINT,
        request_stop
    )

    try:

        for epoch in range(
            start_epoch,
            epochs + 1
        ):

            # Random sentence order
            order = np.random.permutation(
                len(data)
            )

            total_loss = 0.0

            for idx in order:

                sequence = data[idx]

                # ----------------------------------------------------
                # Example:
                #
                # [BOS, i, like, cats, EOS]
                #
                # input:
                #
                # [BOS, i, like, cats]
                #
                # target:
                #
                # [i, like, cats, EOS]
                # ----------------------------------------------------

                input_ids = sequence[:-1]
                target_ids = sequence[1:]

                # ----------------------------------------------------
                # Forward + backward
                # ----------------------------------------------------

                loss, grads = (
                    model.loss_and_backward(
                        input_ids,
                        target_ids
                    )
                )

                # ----------------------------------------------------
                # Parameter update
                # ----------------------------------------------------

                optimizer.step(
                    model.p,
                    grads,
                    clip_norm=1.0
                )

                total_loss += loss

                if stop["requested"]:
                    break

            # --------------------------------------------------------
            # Interrupted in the middle of this epoch
            #
            # 가중치는 마지막 step 까지 반영된 상태로 저장하고,
            # epoch 는 직전까지만 완료로 기록한다.
            # (다시 실행하면 이 epoch 부터 이어서 학습)
            # --------------------------------------------------------

            if stop["requested"]:

                if state_path is not None:

                    save_state(
                        state_path,
                        model,
                        optimizer,
                        epoch - 1,
                        sentences
                    )

                    print(
                        f"epoch {epoch} 진행 중 중단 -> "
                        f"{Path(state_path).name} 에 저장했습니다."
                    )

                return False

            # --------------------------------------------------------
            # Training state (every epoch)
            # --------------------------------------------------------

            if state_path is not None:

                save_state(
                    state_path,
                    model,
                    optimizer,
                    epoch,
                    sentences
                )

            # --------------------------------------------------------
            # Epoch checkpoint
            # --------------------------------------------------------

            if (
                checkpoint_dir is not None
                and
                epoch % checkpoint_every == 0
            ):

                checkpoint_path = (
                    checkpoint_dir
                    / f"epoch_{epoch:04d}.npz"
                )

                np.savez_compressed(
                    checkpoint_path,
                    **model.p
                )

            # --------------------------------------------------------
            # Print loss
            # --------------------------------------------------------

            if (
                epoch == start_epoch
                or
                epoch % print_every == 0
            ):

                average_loss = (
                    total_loss
                    /
                    len(data)
                )

                print(
                    f"epoch {epoch:4d} "
                    f"| loss {average_loss:.4f}"
                )

    finally:

        signal.signal(
            signal.SIGINT,
            previous_handler
        )

    return True


# ====================================================================
# Main
# ====================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="TinyGPT training"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=700,
        help="목표 epoch 수 (저장된 모델이 이보다 적게 학습됐으면 이어서 학습)"
    )

    parser.add_argument(
        "--retrain",
        action="store_true",
        help="저장된 모델을 무시하고 처음부터 다시 학습"
    )

    args = parser.parse_args()

    # ================================================================
    # 1. Training corpus
    # ================================================================

    sentences = [

        "the cat sleeps on the warm sofa every afternoon",
        "the cat chases a small mouse in the kitchen",
        "our cat likes to sit by the window",

        "the dog runs in the park with a red ball",
        "the dog barks loudly when the mailman comes",

        "my brother plays soccer with his friends after school",
        "my sister reads a book in her room at night",
        "my mother makes bread in the kitchen every saturday",
        "my father reads the newspaper after breakfast",

        "i drink a cup of coffee every morning",
        "i walk to school with my best friend",
        "i like to read books on a rainy day",

        "we eat dinner together at seven every evening",
        "we go to the beach in the summer",
        "we like to watch the stars at night",

        "she cooks pasta for her family on sunday",
        "she listens to music while she studies",

        "he rides his bike to work every day",
        "he watches a movie with his wife on friday",

        "they visit their grandparents every winter",
        "they play board games at home when it rains",

        "the children play in the garden after lunch",
        "the children sing a song in the classroom",

        "the students study hard for the final exam",
        "the students eat lunch in the school cafeteria",

        "the teacher writes a long sentence on the board",
        "the teacher gives us homework every weekend",

        "the little girl draws a picture of her dog",
        "the old man walks slowly along the river",
        "the baby sleeps quietly in a small bed",

        "the farmer grows rice and vegetables in his field",
        "the doctor helps sick people at the hospital",

        "the sun rises early in the summer morning",
        "the rain falls softly on the old roof",
        "the birds sing in the tall trees at dawn",
        "the train leaves the station at eight in the morning",
    ]

    # ================================================================
    # 2. Tokenization
    # ================================================================

    (
        token_to_id,
        id_to_token,
        data
    ) = build_dataset(
        sentences
    )

    # Maximum sequence length
    max_context = max(
        len(sequence) - 1
        for sequence in data
    )

    print(
        "vocab size:",
        len(token_to_id)
    )

    print(
        "max context:",
        max_context
    )

    # ================================================================
    # 3. Create GPT
    # ================================================================

    model = TinyGPT(

        vocab_size=len(
            token_to_id
        ),

        max_context=max_context,

        d_model=64,

        n_heads=4,

        d_ff=128,

        n_layers=2,

        seed=42
    )

    parameter_count = sum(
        parameter.size
        for parameter in model.p.values()
    )

    print(
        "parameters:",
        parameter_count
    )

    optimizer = Adam(
        model.p,
        lr=2e-3
    )

    # ================================================================
    # 4. Load saved state
    #
    # model.npz 가 있고 학습 문장/모델 구조가 같으면 불러온다.
    # ================================================================

    state_path = Path(__file__).with_name(
        "model.npz"
    )

    done_epoch = 0

    if not args.retrain:

        loaded = load_state(
            state_path,
            model,
            optimizer,
            sentences
        )

        if loaded is not None:

            done_epoch = loaded

            print(
                f"저장된 모델을 불러왔습니다: {state_path.name} "
                f"({done_epoch} epoch 학습됨)"
            )

        elif state_path.exists():

            print(
                "학습 문장 또는 모델 구조가 바뀌어 처음부터 학습합니다."
            )

    # ================================================================
    # 5. Train (처음부터 또는 이어서)
    # ================================================================

    if done_epoch >= args.epochs:

        print(
            f"이미 {done_epoch} epoch 까지 학습되어 있습니다. "
            "더 학습하려면 --epochs 를 늘리고, "
            "처음부터 하려면 --retrain 을 붙이세요."
        )

    else:

        print(
            f"학습: epoch {done_epoch + 1} -> {args.epochs} "
            "(Ctrl+C 로 중단해도 가중치는 저장됩니다)"
        )

        completed = train(

            model,

            data,

            epochs=args.epochs,

            print_every=50,

            checkpoint_dir=(
                Path(__file__).with_name(
                    "checkpoints_v2"
                )
            ),

            checkpoint_every=100,

            optimizer=optimizer,

            start_epoch=done_epoch + 1,

            state_path=state_path,

            sentences=sentences
        )

        if not completed:

            print(
                "학습이 중단되었습니다. "
                "다시 실행하면 이어서 학습합니다."
            )

    print()
    print(
        f"저장 위치: {state_path}"
    )

    print(
        "문장 생성: python generate.py"
    )
