# TinyGPT — NumPy로 구현한 GPT

[English](README.md) | **한국어**

> [!IMPORTANT]
> **📐 [GPT math.pdf](GPT%20math.pdf) — 모든 순전파·역전파 공식을 손으로 유도한 노트 (22쪽)**  
> NumPy 코드는 이 노트를 바탕으로 구현했고, 한 줄씩 대조해 검증했습니다.

[라이브 데모](https://columnback.github.io/tinygpt-numpy/demo.html#ko) | [비주얼 가이드](https://columnback.github.io/tinygpt-numpy/#ko)

모델 구현에 PyTorch나 TensorFlow를 쓰지 않고 **NumPy만으로** 구현한 GPT(디코더 전용 Transformer)입니다.

이 프로젝트는 GPT의 수학적 구조를 이해하고, **구현이 수학적 유도 과정을 그대로 따르는지 검증하는 것**에 초점을 둡니다.

수학적 유도는 **손으로 직접 풀어 가며 단계별로 공부했습니다** ([`GPT math.pdf`](GPT%20math.pdf) 참고). 이 수식을 바탕으로 **ChatGPT를 사용해 수식을 NumPy 코드로 구현했습니다**.

**그다음, 완성된 코드를 손으로 쓴 유도 과정과 꼼꼼히 대조하여 의도한 수학이 올바르게 반영되었는지 검증했습니다.** 검증 범위에는 텐서 차원, 순전파 계산, 역전파, 중간 그래디언트, 전체 그래디언트 흐름이 포함됩니다.

구현은 서로 독립적인 두 가지 방법으로 검증했습니다. 그래디언트 검증 스크립트 역시 **ChatGPT**를 사용해 작성했습니다.

- 중앙 차분을 이용한 **수치 그래디언트 검증** (`gradcheck.py`)
- 독립적인 기준으로서의 **TensorFlow 자동 미분** (`tf_gradcheck.py`)

- 손글씨 유도 노트: [`GPT math.pdf`](GPT%20math.pdf) (22쪽)
- 학습된 가중치 포함: 클론 직후 바로 `python generate.py` 실행 가능

```
> the dog
  - the dog runs in the park with a red ball
  - the dog barks loudly when the mailman comes
```

## 파일 구성

| 파일 | 역할 |
|---|---|
| [`tinygpt.py`](tinygpt.py) | 모델: `TinyGPT` (순전파/역전파), `Adam`, 토크나이저, 가중치 저장/불러오기 |
| [`train.py`](train.py) | 학습: 코퍼스 + 학습 루프 (중단 후 재개 지원) |
| [`generate.py`](generate.py) | 추론: 단어 한두 개를 입력하면 문장 생성 |
| [`gradcheck.py`](gradcheck.py) | 그래디언트 검증 ① 수치 그래디언트 (차트가 포함된 HTML 리포트 생성) |
| [`tf_gradcheck.py`](tf_gradcheck.py) | 그래디언트 검증 ② TensorFlow `GradientTape` |
| `model.npz` | 학습된 가중치 (700 에폭) |
| [`export_web.py`](export_web.py) | 브라우저 데모용으로 `model.npz`를 `docs/model.js`로 내보냄 (재학습 후 다시 실행) |
| [`docs/`](docs) | 비주얼 가이드와 라이브 데모 (GitHub Pages). `tinygpt-web.js`는 순전파를 JavaScript로 옮긴 것 |
| `GPT math.pdf` | 순전파/역전파 유도 노트 |

## 빠른 시작

```bash
pip install -r requirements.txt     # numpy만 필요

python generate.py                  # 포함된 model.npz로 문장 생성
python train.py                     # 직접 학습 (~1분)
```

### 문장 생성 — `generate.py`

학습 문장 36개와 어휘 목록을 출력한 뒤 입력을 기다립니다.
문장을 시작하는 단어(`the`, `my`, `we`, `she`, …)가 가장 잘 동작합니다.

```bash
python generate.py                         # 기본값: temperature 0.7, 샘플 3개
python generate.py --temperature 1.0 --samples 5
```

- 학습 문장에 없는 단어는 알려 줍니다 (예: `hello`)
- 각 출력에 학습 문장인지, 새로운 조합인지 표시합니다
- 모델 구조와 어휘는 `model.npz`에서 복원되므로 별도 설정이 필요 없습니다

### 학습 — `train.py`

```bash
python train.py                 # 300 에폭까지 학습 (이미 학습되어 있으면 건너뜀)
python train.py --epochs 1000   # 저장된 모델을 1000 에폭까지 이어서 학습
python train.py --retrain       # 처음부터 다시 학습
```

- **매 에폭마다 `model.npz`에 저장** — 재부팅해도 가중치가 남습니다
- **Ctrl+C**를 누르면 현재 스텝을 마치고 저장한 뒤 멈춥니다
- **다시 실행하면 이어서 학습** — 파라미터와 함께 Adam 상태(m, v, t)도 저장하므로 정확히 멈춘 지점부터 재개합니다
- 임시 파일에 저장한 뒤 교체하므로, 저장 도중 비정상 종료되어도 파일이 손상되지 않습니다
- **학습 문장을 바꾸면** 다음 실행 때 자동으로 처음부터 다시 학습합니다
- 100 에폭마다 `checkpoints_v2/`에 스냅샷도 저장합니다

코퍼스를 바꾸려면 `train.py`의 `sentences` 리스트를 수정하세요.

## 아키텍처

Post-LayerNorm을 사용하는 GPT 스타일 디코더 전용 Transformer입니다.

```
token ids ──> E[token] + P[position]            토큰 + 학습되는 위치 임베딩
                 │
          ┌──────┴───────── × N blocks ─────────────────────┐
          │  O = Concat(head_1..head_H) W_O                 │  마스크드 멀티헤드 셀프 어텐션
          │  Y = LayerNorm(X + O)                           │  잔차 연결 + LN
          │  F = GELU(Y W_1 + b_1) W_2 + b_2                │  피드포워드
          │  X = LayerNorm(Y + F)                           │  잔차 연결 + LN
          └──────┬──────────────────────────────────────────┘
                 │
          logits = X Eᵀ                                   가중치 공유 (W_LM = Eᵀ)
          loss   = cross-entropy(softmax(logits), next token)
```

각 헤드:

$$Q = XW_Q,\quad K = XW_K,\quad V = XW_V,\qquad A = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}} + M\right),\qquad O_h = AV$$

($M$은 미래 위치를 가리는 causal mask입니다.)

| 설정 | 값 |
|---|---|
| 어휘 (단어 단위) | 169개 (`<BOS>`, `<EOS>`, `<UNK>` 포함) |
| d_model / heads / d_ff / layers | 64 / 4 / 128 / 2 |
| 최대 컨텍스트 | 11 토큰 |
| 파라미터 수 | 77,952 |
| 옵티마이저 | Adam (lr 2e-3, 그래디언트 클리핑 1.0) |
| 배치 | 문장 1개 (문장마다 가중치 업데이트, 에폭당 36회 업데이트) |

## 역전파

역전파 공식은 **손으로 직접 풀어 가며 단계별로 공부했습니다**. 이 유도를 바탕으로 **ChatGPT를 사용해 역전파를 NumPy로 구현했습니다**.

**그다음, 구현을 수학적 유도와 꼼꼼히 대조하여 의도한 수학이 올바르게 반영되었는지 검증했습니다.** 유도 과정은 [`GPT math.pdf`](GPT%20math.pdf)에 있습니다.

| 쪽 | 내용 |
|---|---|
| 1–5 | 순전파: 임베딩, 어텐션, LayerNorm, FFN, LM 헤드, 크로스 엔트로피 |
| 6 | Softmax + 크로스 엔트로피 역전파 → $G_L = \frac{1}{T}(P - Y)$ |
| 7 | LM 헤드 역전파, 가중치 공유 ($G_E = G_E^{\text{emb}} + G_E^{\text{LM}}$) |
| 7–11 | LayerNorm 역전파 (경로별) |
| 12–14 | 간결한 LayerNorm 역전파 (코드에서 쓰는 단순화된 형태) |
| 15 | 잔차 연결, FFN 역전파, GELU 역전파 |
| 16–19 | 어텐션 역전파: $W_O$, $AV$, softmax, $QK^\top$, $W_Q, W_K, W_V$ |
| 20 | 임베딩 역전파 (반복된 토큰의 그래디언트는 누적) |

코드에 쓰인 핵심 공식:

$$G_S = A \odot \left[G_A - (G_A \odot A)\mathbf{1}\mathbf{1}^\top\right] \qquad \text{(softmax 역전파)}$$

$$G_R = \frac{1}{D} \odot \left[G_{\hat R} - \mathrm{mean}(G_{\hat R}) - \hat R \odot \mathrm{mean}(G_{\hat R} \odot \hat R)\right],\quad G_{\hat R} = G_Y \odot \gamma \qquad \text{(LayerNorm 역전파)}$$

## 수식 검증

수학적 유도와 NumPy 구현을 서로 독립적인 두 가지 방법으로 확인합니다.
먼저 구성 요소(GELU, LayerNorm, 어텐션 헤드, 블록)를 확인한 다음, 전체 모델의 모든 파라미터를 확인합니다.
γ 누락 같은 버그가 숨지 못하도록 LayerNorm의 γ, β와 편향을 기본값(1과 0)에서 일부러 벗어나게 설정합니다.

### ① 수치 그래디언트 — `gradcheck.py`

수치 그래디언트 검증 스크립트는 **ChatGPT**를 사용해 작성했으며, 중앙 차분 $\frac{L(\theta+h) - L(\theta-h)}{2h}$와 비교합니다.

```bash
python gradcheck.py              # 결과 출력 + 차트가 포함된 HTML 리포트 열기
python gradcheck.py --no-open    # 터미널 출력만
```

| 대상 | 상대 오차 |
|---|---|
| GELU, LayerNorm, 어텐션 헤드, 블록 | ~1e-11 |
| 전체 모델 (파라미터 그룹 32개) | ~1e-7 |
| **결과** | **56 / 56 PASS** |

판정 기준: 상대 오차 < 1e-6 이면 PASS, 1e-6 ~ 1e-4 이면 WARN, 1e-4 이상이면 FAIL.

리포트(`gradcheck_report.html`)에는 검증별 오차, 해석적 그래디언트 대 수치 그래디언트 산점도
(모든 점이 y = x 위에 있어야 함), 스텝 크기 h에 따른 오차가 표시됩니다.

> 전체 모델이 ~1e-7에서 멈추는 이유: 손실의 `log(p + 1e-12)` 안에 있는 `1e-12`가 확률이 매우 작은 토큰에서
> ~1e-7 정도의 차이를 만듭니다. 수식 오류가 아닙니다.

### ② TensorFlow 자동 미분 — `tf_gradcheck.py`

TensorFlow 그래디언트 검증 스크립트도 **ChatGPT**를 사용해 작성했습니다. 같은 모델을 TensorFlow 연산
(`tf.nn.gelu`, `tf.nn.softmax`, `sparse_softmax_cross_entropy_with_logits`)으로 **독립적으로** 다시 구현하고 `GradientTape` 결과와 비교합니다.

```bash
pip install tensorflow           # Python 3.13 이하 필요
python tf_gradcheck.py
python tf_gradcheck.py --model model.npz    # 학습된 가중치(d_model 64)로 검증
```

| 대상 | 상대 오차 |
|---|---|
| LayerNorm, 어텐션 헤드 | ~1e-16 (정확히 일치) |
| GELU, 블록, 전체 모델 | ~1e-10 |
| **결과** | **147 / 147 PASS** |

> GELU가 들어간 항목이 1e-10인 이유: `tf.nn.gelu`는 순전파에서 이미 ~1e-10 정도 차이가 납니다 (TF 구현상의 세부 사항).
> 같은 GELU 공식을 TF로 직접 작성하면 147개 검증 모두 **1e-15** (float64 정밀도)까지 일치합니다.
> `--model`(학습된 가중치)을 쓰면 softmax가 더 뾰족해져 이 차이가 조금 커지므로, WARN(~1e-8)이 몇 개 나올 수 있습니다 (FAIL은 없음).

## 한계

- **대부분 학습 문장을 외웁니다.** 문장이 36개뿐이고 대부분의 단어가 한 번씩만 등장하므로, 출력 대부분이 학습 문장 그대로입니다.
  일부 위치에는 정답이 여러 개이기 때문에 손실이 0.42 근처에서 더 내려가지 않습니다 (예: `the` 다음에 올 수 있는 문장이 20개가 넘음).
- **단어 단위 토큰화**이므로 학습 문장에 없는 단어는 처리할 수 없습니다.
- 교육용 구현입니다: 배치 없이 한 번에 문장 하나씩, CPU에서 NumPy로 계산합니다.
- Post-LayerNorm 구조입니다 (GPT-2 이후는 Pre-LayerNorm 사용).

## 요구 사항

- Python 3.10+ (3.14에서 테스트)
- NumPy (2.3에서 테스트)
- TensorFlow는 `tf_gradcheck.py`에만 필요 (Python 3.13 + TensorFlow 2.21에서 테스트)
