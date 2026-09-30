# TinyGPT — A GPT implemented with NumPy

A GPT (decoder-only Transformer) implemented using **NumPy only** — without PyTorch or TensorFlow for the model implementation.

This project focuses on understanding the mathematical structure of a GPT and **verifying that the implementation faithfully follows the corresponding mathematical derivations**.

The mathematical derivations were **worked through by hand and studied step by step** (See [`GPT math.pdf`](GPT%20math.pdf)). Based on these mathematical formulations, **ChatGPT was used to implement the mathematics in NumPy code**.

**The resulting code was then carefully compared against the handwritten mathematical derivations to verify that the implementation correctly reflected the intended mathematics.** This verification included tensor dimensions, forward computations, backward propagation, intermediate gradients, and overall gradient flow.

The implementation was independently validated in two ways. The gradient-checking scripts were also developed with **ChatGPT**:

- **Numerical gradient checking** using central finite differences (`gradcheck.py`)
- **TensorFlow automatic differentiation** as an independent reference (`tf_gradcheck.py`)

- Handwritten derivation notes: [`GPT math.pdf`](GPT%20math.pdf) (22 pages)
- Trained weights included: run `python generate.py` right after cloning

```
> the dog
  - the dog runs in the park with a red ball
  - the dog barks loudly when the mailman comes
```

## Files

| File | Role |
|---|---|
| [`tinygpt.py`](tinygpt.py) | The model: `TinyGPT` (forward/backward), `Adam`, tokenizer, saving/loading weights |
| [`train.py`](train.py) | Training: corpus + training loop (interrupt & resume supported) |
| [`generate.py`](generate.py) | Inference: type one or two words, get a sentence |
| [`gradcheck.py`](gradcheck.py) | Gradient check ① numerical gradient (generates an HTML report with charts) |
| [`tf_gradcheck.py`](tf_gradcheck.py) | Gradient check ② TensorFlow `GradientTape` |
| `model.npz` | Trained weights (700 epochs) |
| `GPT math.pdf` | Forward/backward derivation notes |

## Quick start

```bash
pip install -r requirements.txt     # only numpy

python generate.py                  # generate sentences with the included model.npz
python train.py                     # train it yourself (~1 min)
```

### Generating sentences — `generate.py`

It prints the 36 training sentences and the vocabulary, then waits for input.
Words that start a sentence (`the`, `my`, `we`, `she`, …) work best.

```bash
python generate.py                         # default: temperature 0.7, 3 samples
python generate.py --temperature 1.0 --samples 5
```

- Words that are not in the training sentences are reported (e.g. `hello`)
- Each output is tagged as a training sentence or a new combination
- The model architecture and vocabulary are restored from `model.npz`, so no configuration is needed

### Training — `train.py`

```bash
python train.py                 # train up to 300 epochs (skipped if already trained)
python train.py --epochs 1000   # continue the saved model up to 1000 epochs
python train.py --retrain       # start over from scratch
```

- **Saves to `model.npz` after every epoch** — the weights survive a reboot
- **Ctrl+C** finishes the current step, saves, and stops
- **Re-running resumes training** — the Adam state (m, v, t) is saved along with the parameters, so it picks up exactly where it left off
- Saves go to a temp file that is then swapped in, so a crash mid-save cannot corrupt the file
- **If you change the training sentences**, the next run retrains from scratch automatically
- A snapshot is also written to `checkpoints_v2/` every 100 epochs

Edit the `sentences` list in `train.py` to change the corpus.

## Architecture

GPT-style decoder-only Transformer with Post-LayerNorm.

```
token ids ──> E[token] + P[position]            token + learned positional embedding
                 │
          ┌──────┴───────── × N blocks ─────────────────────┐
          │  O = Concat(head_1..head_H) W_O                 │  masked multi-head self-attention
          │  Y = LayerNorm(X + O)                           │  residual + LN
          │  F = GELU(Y W_1 + b_1) W_2 + b_2                │  feed-forward
          │  X = LayerNorm(Y + F)                           │  residual + LN
          └──────┬──────────────────────────────────────────┘
                 │
          logits = X Eᵀ                                   weight tying (W_LM = Eᵀ)
          loss   = cross-entropy(softmax(logits), next token)
```

Each head:

$$Q = XW_Q,\quad K = XW_K,\quad V = XW_V,\qquad A = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}} + M\right),\qquad O_h = AV$$

($M$ is the causal mask that blocks future positions.)

| Setting | Value |
|---|---|
| Vocabulary (word-level) | 169 (including `<BOS>`, `<EOS>`, `<UNK>`) |
| d_model / heads / d_ff / layers | 64 / 4 / 128 / 2 |
| Max context | 11 tokens |
| Parameters | 77,952 |
| Optimizer | Adam (lr 2e-3, gradient clipping 1.0) |
| Batch | 1 sentence (weights updated after every sentence, 36 updates per epoch) |

## Backpropagation

The backward formulas were **worked through by hand and studied step by step**. Based on these derivations, **ChatGPT was used to implement the backward pass in NumPy**.

**The resulting implementation was then carefully compared against the mathematical derivations to verify that it correctly reflected the intended mathematics.** The derivations are in [`GPT math.pdf`](GPT%20math.pdf).

| Pages | Contents |
|---|---|
| 1–5 | Forward: embedding, attention, LayerNorm, FFN, LM head, cross-entropy |
| 6 | Softmax + cross-entropy backward → $G_L = \frac{1}{T}(P - Y)$ |
| 7 | LM head backward, weight tying ($G_E = G_E^{\text{emb}} + G_E^{\text{LM}}$) |
| 7–11 | LayerNorm backward (path by path) |
| 12–14 | Compact LayerNorm backward (the simplified form used in the code) |
| 15 | Residual, FFN backward, GELU backward |
| 16–19 | Attention backward: $W_O$, $AV$, softmax, $QK^\top$, $W_Q, W_K, W_V$ |
| 20 | Embedding backward (gradients of repeated tokens are accumulated) |

Key formulas used in the code:

$$G_S = A \odot \left[G_A - (G_A \odot A)\mathbf{1}\mathbf{1}^\top\right] \qquad \text{(softmax backward)}$$

$$G_R = \frac{1}{D} \odot \left[G_{\hat R} - \mathrm{mean}(G_{\hat R}) - \hat R \odot \mathrm{mean}(G_{\hat R} \odot \hat R)\right],\quad G_{\hat R} = G_Y \odot \gamma \qquad \text{(LayerNorm backward)}$$

## Verifying the formulas

The mathematical derivations and their NumPy implementation are checked two independent ways.
Components (GELU, LayerNorm, attention head, block) are checked first, then every parameter of the full model.
LayerNorm γ, β and the biases are deliberately perturbed away from their defaults (1 and 0) so that bugs like a missing γ cannot hide.

### ① Numerical gradient — `gradcheck.py`

The numerical gradient-checking script was developed with **ChatGPT** and compares against the central difference $\frac{L(\theta+h) - L(\theta-h)}{2h}$.

```bash
python gradcheck.py              # print results + open an HTML report with charts
python gradcheck.py --no-open    # terminal only
```

| Target | Relative error |
|---|---|
| GELU, LayerNorm, attention head, block | ~1e-11 |
| Full model (32 parameter groups) | ~1e-7 |
| **Result** | **56 / 56 PASS** |

The report (`gradcheck_report.html`) shows per-check errors, an analytic-vs-numerical scatter plot
(every point should sit on y = x), and the error as a function of the step size h.

> Why the full model stops at ~1e-7: the `1e-12` inside the loss's `log(p + 1e-12)` causes a ~1e-7 difference
> for tokens with very small probability. It is not a formula error.

### ② TensorFlow autodiff — `tf_gradcheck.py`

The TensorFlow gradient-checking script was also developed with **ChatGPT**. The same model is re-implemented **independently** with TensorFlow ops
(`tf.nn.gelu`, `tf.nn.softmax`, `sparse_softmax_cross_entropy_with_logits`) and compared against `GradientTape`.

```bash
pip install tensorflow           # requires Python 3.13 or lower
python tf_gradcheck.py
python tf_gradcheck.py --model model.npz    # check with the trained weights (d_model 64)
```

| Target | Relative error |
|---|---|
| LayerNorm, attention head | ~1e-16 (exact match) |
| GELU, block, full model | ~1e-10 |
| **Result** | **147 / 147 PASS** |

> Why anything involving GELU is at 1e-10: `tf.nn.gelu` already differs by ~1e-10 in the forward pass (TF implementation detail).
> Writing the same GELU formula in TF by hand makes all 147 checks agree to **1e-15** (float64 precision).
> With `--model` (trained weights), sharper softmax amplifies this slightly, so a few WARNs (~1e-8) may appear (no FAILs).

## Limitations

- **It mostly memorizes the training sentences.** There are only 36 sentences and most words appear once, so most outputs are exact training sentences.
  The loss plateaus around 0.42 because some positions have many valid answers (e.g. more than 20 sentences can follow `the`).
- **Word-level tokenization**, so words outside the training sentences cannot be handled.
- Educational implementation: no batching, one sentence at a time, NumPy on the CPU.
- Post-LayerNorm (GPT-2 and later use Pre-LayerNorm).

## Requirements

- Python 3.10+ (tested on 3.14)
- NumPy (tested on 2.3)
- TensorFlow only for `tf_gradcheck.py` (tested on Python 3.13 + TensorFlow 2.21)
