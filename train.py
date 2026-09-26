"""
TinyGPT training

Trains TinyGPT on the training sentences and saves it to model.npz.

- Saves to model.npz after every epoch (survives a reboot)
- Ctrl+C saves the weights up to the current step
- Re-running resumes from the epoch after the saved one
- Retrains from scratch if the sentences or architecture change

Usage:

    python train.py
    python train.py --epochs 500     # continue the saved model up to 500 epochs
    python train.py --retrain        # start over from scratch

Sentence generation (inference): generate.py
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
    # If state_path is given, the training state is saved after every epoch.
    #
    # Ctrl+C finishes the current step, saves, and stops.
    # (Press it again to force quit; the last epoch's state is still kept.)
    #
    # Returns True if training finished, False if it was interrupted.
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
            "\nStop requested: finishing the current step and saving. "
            "(Force quit: press Ctrl+C again)"
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
            # Save the weights including the last step,
            # but record only the previous epoch as completed.
            # (the next run resumes from this epoch)
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
                        f"Interrupted during epoch {epoch} -> "
                        f"saved to {Path(state_path).name}."
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
        help="target number of epochs (continues if the saved model has fewer)"
    )

    parser.add_argument(
        "--retrain",
        action="store_true",
        help="ignore the saved model and train from scratch"
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
    # Load model.npz if it exists and matches the sentences / architecture.
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
                f"Loaded saved model: {state_path.name} "
                f"({done_epoch} epochs trained)"
            )

        elif state_path.exists():

            print(
                "Training sentences or architecture changed; training from scratch."
            )

    # ================================================================
    # 5. Train (from scratch or resume)
    # ================================================================

    if done_epoch >= args.epochs:

        print(
            f"Already trained for {done_epoch} epochs. "
            "To train more, raise --epochs; "
            "to start over, add --retrain."
        )

    else:

        print(
            f"Training: epoch {done_epoch + 1} -> {args.epochs} "
            "(weights are saved even if you press Ctrl+C)"
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
                "Training was interrupted. "
                "Run again to resume."
            )

    print()
    print(
        f"Saved to: {state_path}"
    )

    print(
        "Generate sentences: python generate.py"
    )
