# TinyGPT — NumPy로 처음부터 만든 GPT

PyTorch·TensorFlow 없이 **NumPy만으로** GPT(Transformer decoder)를 구현한 프로젝트입니다.
forward뿐 아니라 **backward(역전파) 수식을 모두 손으로 유도해서 직접 코딩**했고,
그 수식이 맞는지 **numerical gradient**와 **TensorFlow 자동미분** 두 가지 방법으로 검증했습니다.

- 손으로 유도한 수식 노트: [`GPT math.pdf`](GPT%20math.pdf) (22쪽)
- 학습된 가중치 포함: 받자마자 `python generate.py`로 문장 생성 가능

```
> the dog
  - the dog runs in the park with a red ball
  - the dog barks loudly when the mailman comes
```

## 파일 구성

| 파일 | 역할 |
|---|---|
| [`tinygpt.py`](tinygpt.py) | 모델 본체: `TinyGPT`(forward/backward), `Adam`, 토큰화, 가중치 저장·불러오기 |
| [`train.py`](train.py) | 학습: 학습 문장 + 학습 루프 (중단·재개 지원) |
| [`generate.py`](generate.py) | 추론: 한두 단어를 입력하면 문장 생성 |
| [`gradcheck.py`](gradcheck.py) | 수식 검증 ① numerical gradient (차트가 있는 HTML 리포트 생성) |
| [`tf_gradcheck.py`](tf_gradcheck.py) | 수식 검증 ② TensorFlow `GradientTape` |
| `model.npz` | 학습된 가중치 (700 epoch) |
| `GPT math.pdf` | forward·backward 수식 유도 노트 |

## 빠른 시작

```bash
pip install -r requirements.txt     # numpy 만 필요

python generate.py                  # 포함된 model.npz 로 바로 문장 생성
python train.py                     # 직접 학습 (약 1분)
```

### 문장 생성 — `generate.py`

실행하면 학습 문장 36개와 단어 목록을 보여주고, 입력을 기다립니다.
문장을 시작하는 단어(`the`, `my`, `we`, `she` …)로 입력하면 가장 자연스럽습니다.

```bash
python generate.py                         # 기본: temperature 0.7, 3문장
python generate.py --temperature 1.0 --samples 5
```

- 학습 문장에 없는 단어는 알려줍니다 (`hello` → 학습 문장에 없는 단어입니다)
- 생성된 문장이 학습 문장 그대로인지 `[학습 문장]` / `[새 조합]`으로 표시합니다
- 모델 구조와 단어장은 `model.npz`에서 자동으로 복원하므로 설정이 필요 없습니다

### 학습 — `train.py`

```bash
python train.py                 # 300 epoch 까지 학습 (이미 학습돼 있으면 건너뜀)
python train.py --epochs 1000   # 저장된 모델을 1000 epoch 까지 이어서 학습
python train.py --retrain       # 처음부터 다시 학습
```

- **매 epoch 끝마다 `model.npz`에 저장** — 컴퓨터를 껐다 켜도 가중치가 남습니다
- **Ctrl+C로 중단**하면 진행 중인 step까지 반영해서 저장합니다
- **다시 실행하면 이어서 학습** — parameter와 함께 Adam 상태(m, v, t)도 저장하기 때문에 끊긴 지점부터 정확히 재개됩니다
- 저장은 임시 파일에 쓴 뒤 교체하는 방식이라 저장 도중 꺼져도 파일이 깨지지 않습니다
- **학습 문장을 바꾸면** 다음 실행 때 자동으로 처음부터 다시 학습합니다
- 100 epoch마다 `checkpoints_v2/`에 스냅샷도 남깁니다

학습 문장은 `train.py`의 `sentences` 목록에서 바꿀 수 있습니다.

## 모델 구조

GPT-style decoder-only Transformer, Post-LayerNorm.

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

각 head:

$$Q = XW_Q,\quad K = XW_K,\quad V = XW_V,\qquad A = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}} + M\right),\qquad O_h = AV$$

($M$은 미래 위치를 막는 causal mask)

| 설정 | 값 |
|---|---|
| 단어장 (word-level) | 169개 (`<BOS>`, `<EOS>`, `<UNK>` 포함) |
| d_model / heads / d_ff / layers | 64 / 4 / 128 / 2 |
| 최대 context | 11 tokens |
| parameters | 77,952 |
| optimizer | Adam (lr 2e-3, gradient clipping 1.0) |
| batch | 문장 1개 (문장마다 가중치 업데이트, 36회/epoch) |

## 역전파 수식

모든 gradient를 자동미분 없이 직접 유도해서 구현했습니다. 유도 과정은 [`GPT math.pdf`](GPT%20math.pdf)에 있습니다.

| 쪽 | 내용 |
|---|---|
| 1–5 | Forward: embedding, attention, LayerNorm, FFN, LM head, cross-entropy |
| 6 | Softmax + cross-entropy backward → $G_L = \frac{1}{T}(P - Y)$ |
| 7 | LM head backward, weight tying ($G_E = G_E^{\text{emb}} + G_E^{\text{LM}}$) |
| 7–11 | LayerNorm backward (경로별 유도) |
| 12–14 | Compact LayerNorm backward (구현에 쓴 간단한 형태) |
| 15 | Residual, FFN backward, GELU backward |
| 16–19 | Attention backward: $W_O$, $AV$, softmax, $QK^\top$, $W_Q, W_K, W_V$ |
| 20 | Embedding backward (같은 token의 gradient 누적) |

구현에 사용한 대표 수식:

$$G_S = A \odot \left[G_A - (G_A \odot A)\mathbf{1}\mathbf{1}^\top\right] \qquad \text{(softmax backward)}$$

$$G_R = \frac{1}{D} \odot \left[G_{\hat R} - \mathrm{mean}(G_{\hat R}) - \hat R \odot \mathrm{mean}(G_{\hat R} \odot \hat R)\right],\quad G_{\hat R} = G_Y \odot \gamma \qquad \text{(LayerNorm backward)}$$

## 수식 검증

손으로 유도한 backward가 맞는지 두 가지 독립적인 방법으로 확인했습니다.
부품(GELU, LayerNorm, attention head, block) 단위로 먼저 확인하고, 전체 모델의 모든 parameter를 확인합니다.
LayerNorm의 γ, β와 bias는 기본값(1, 0)에서 일부러 흔들어서, γ를 빠뜨리는 류의 버그도 잡히게 했습니다.

### ① Numerical gradient — `gradcheck.py`

중앙차분 $\frac{L(\theta+h) - L(\theta-h)}{2h}$와 비교합니다.

```bash
python gradcheck.py              # 결과 출력 + HTML 리포트(차트)를 브라우저로 열기
python gradcheck.py --no-open    # 터미널 출력만
```

| 대상 | relative error |
|---|---|
| GELU, LayerNorm, attention head, block | ~1e-11 |
| 전체 모델 (32개 parameter 그룹) | ~1e-7 |
| **결과** | **56 / 56 PASS** |

리포트(`gradcheck_report.html`)에는 항목별 오차, analytic vs numerical 산점도(모든 점이 y = x 위에 놓이는지),
step size h에 따른 오차 곡선이 들어 있습니다.

> 전체 모델이 1e-7에서 멈추는 이유: loss의 `log(p + 1e-12)`에 들어간 `1e-12`가
> 확률이 아주 작은 token에서 ~1e-7 정도의 차이를 만들기 때문이며, 수식 오류가 아닙니다.

### ② TensorFlow 자동미분 — `tf_gradcheck.py`

같은 모델을 TensorFlow 연산(`tf.nn.gelu`, `tf.nn.softmax`, `sparse_softmax_cross_entropy_with_logits`)으로
**독립적으로** 다시 구현하고, `GradientTape` 결과와 비교합니다.

```bash
pip install tensorflow           # Python 3.13 이하 필요
python tf_gradcheck.py
python tf_gradcheck.py --model model.npz    # 학습된 가중치(d_model 64)로 검사
```

| 대상 | relative error |
|---|---|
| LayerNorm, attention head | ~1e-16 (완전 일치) |
| GELU, block, 전체 모델 | ~1e-10 |
| **결과** | **147 / 147 PASS** |

> GELU가 들어간 항목이 1e-10인 이유: `tf.nn.gelu`는 forward 값부터 ~1e-10 다릅니다(TF 내부 구현 차이).
> 같은 GELU 식을 TF로 직접 쓰면 147개 전부 **1e-15**(float64 한계)로 일치합니다.
> `--model`로 학습된 가중치를 쓰면 softmax가 날카로워져 이 차이가 조금 커지고, WARN(~1e-8)이 몇 개 나올 수 있습니다(FAIL 없음).

## 한계

- **학습 문장을 거의 외웁니다.** 문장이 36개뿐이고 대부분 단어가 한 번씩만 나와서, 생성 결과가 대부분 학습 문장 그대로입니다.
  loss가 ~0.42에서 멈추는 것은 `the` 뒤에 올 수 있는 문장이 20개 이상인 것처럼 정답이 여러 개인 위치가 있기 때문입니다.
- **단어 단위 토큰화**라서 학습 문장에 없는 단어는 처리하지 못합니다.
- 교육용 구현이라 batch 처리 없이 문장 하나씩, CPU에서 NumPy로 계산합니다.
- Post-LayerNorm 구조입니다 (GPT-2 이후는 Pre-LayerNorm).

## 요구 사항

- Python 3.10+ (3.14에서 테스트)
- NumPy (2.3에서 테스트)
- `tf_gradcheck.py`만 TensorFlow 필요 (Python 3.13 + TensorFlow 2.21에서 테스트)
