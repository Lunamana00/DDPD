# DDPD 연구 진행 A to Z

Updated: 2026-06-21

이 문서는 현재 프로젝트를 처음 보는 사람이 연구 흐름을 따라갈 수 있도록, 지금까지 진행한 작업을 문제 정의부터 데이터셋, 모델 구조, 학습/평가, ablation, 데모, 한계와 다음 단계까지 한 번에 정리한 것이다.

---

## 1. 연구가 출발한 문제

처음 관심사는 단순한 게임 AI 길찾기가 아니라, **플레이어의 1인칭 시각 경험과 최근 움직임을 바탕으로 앞으로의 local path를 예측할 수 있는가**였다.

핵심 가정은 두 가지였다.

1. 플레이어는 화면에서 환경 단서를 본다.
   - 벽, 복도, 열린 공간, 장애물, 적, 문처럼 보이는 구조, 바닥/조명/텍스처 등의 시각 정보가 이동 판단에 영향을 준다.

2. 경험은 시간에 따라 누적된다.
   - 한 장면의 단일 프레임보다, 최근 1초 동안의 화면 변화와 ego-motion이 앞으로의 경로를 더 잘 설명한다.
   - 또한 기억해야 할 단서는 단순히 시간순 평균으로 사라지는 것이 아니라, 중요한 cue로 남아야 한다.

그래서 최종 연구 질문은 다음처럼 정리되었다.

> 1초 동안의 1인칭 RGB history와 relative ego-motion history만으로, 앞으로의 egocentric local trajectory를 얼마나 잘 예측할 수 있는가?

이 문제는 route ID classification, next action prediction, occupancy prediction과 다르다. 현재 supervised target은 연속적인 future local path다.

```text
Input:
  RGB history                  [B, T, 3, H_img, W_img]
  relative ego-motion history  [B, T, 3]

Target:
  future local path            [B, H, 2]

Output coordinate:
  [forward, right] in current egocentric frame
```

여기서:

- `B`: batch size
- `T`: 과거 history frame 수. 현재 기본값은 1초 × 5 FPS = 5
- `H`: 미래 waypoint 수. 1s=5, 3s=15, 5s=25, 10s=50
- `2`: 현재 pose 기준 local coordinate `[forward, right]`

---

## 2. 초기 프로토타입: ViZDoom local path prediction

초기에는 작은 ViZDoom dataset과 `small_cnn` 기반 모델로 시작했다. 이 단계의 목적은 큰 성능보다 다음 파이프라인이 가능한지 확인하는 것이었다.

1. ViZDoom에서 RGB frame, pose, action, reward, episode metadata를 수집한다.
2. raw episode를 sliding window sample로 변환한다.
3. 과거 1초 history에서 미래 local path를 만든다.
4. constant-velocity baseline과 작은 learned model을 비교한다.

초기 문서 기준 mini result는 다음 방향성을 보여줬다.

| Model | ADE | FDE | 해석 |
|---|---:|---:|---|
| constant velocity | 56.2430 | 96.1106 | 최근 움직임 외삽 |
| ego-motion only | 86.5778 | 144.3438 | ego-motion GRU만으로는 부족 |
| cue-memory direct | 85.2724 | 143.2239 | 직접 회귀는 불안정 |
| cue-memory residual | 51.5144 | 91.0359 | CV prior 위에 residual을 더하는 구조가 더 안정적 |

이때 중요한 전환점은 다음이었다.

> 모델이 모든 경로를 처음부터 직접 맞히게 하지 말고, 최근 움직임 기반 constant-velocity path를 기본값으로 두고 visual cue가 residual을 보정하게 하자.

이후 주력 구조는 `P_pred = P_cv + ΔP_visual` 형태가 되었다.

---

## 3. 데이터셋 설계: WIT-VZ

프로젝트에서 만든 데이터 형식은 WIT-VZ로 정리할 수 있다.

```text
WIT-VZ sample
├─ rgb_history_paths
├─ relative_egomotion_history
├─ future_local_path
├─ future_world_path
├─ current_pose
├─ source/scenario/map/policy metadata
└─ optional cached visual_tokens
```

sample 하나의 핵심 구조는 다음과 같다.

```text
rgb_history:
  [T, 3, 120, 160]

ego_history:
  [T, 3] = [Δforward, Δright, Δyaw]

future_local_path:
  [H, 2] = [forward, right]
```

future path는 global coordinate가 아니라 현재 pose를 원점으로 하는 egocentric coordinate다. 즉 예측값은 “월드 좌표 어디로 가는가”가 아니라 “현재 내가 바라보는 기준에서 앞으로/오른쪽으로 어떻게 움직일 것인가”다.

---

## 4. 데이터셋 확장: v2에서 v4까지

초기 dataset은 작고 scenario 편향이 컸다. 특히 `my_way_home`이 과도하게 많아 source imbalance가 컸다. 그래서 이후 v4 dataset에서는 ViZDoom 기본 scenario 중 안정적으로 실행되는 것들을 최대한 넓게 수집했다.

### v4 default scenario dataset

주요 위치:

```text
data/wit_vz/processed/wit_vz_v4_defaults_001
data/wit_vz/processed/wit_vz_v4_defaults_source_disjoint_001
data/wit_vz/processed/wit_vz_v4_defaults_map_disjoint_001
```

수집 규모:

| 항목 | 값 |
|---|---:|
| runnable ViZDoom scenarios | 15 |
| episodes per scenario | 40 |
| raw episodes | 600 |
| processed supervised samples | 93,403 |
| history | 1s, 5 frames |
| default future target | 1s, 5 waypoints |

episode-disjoint split:

| Split | Samples |
|---|---:|
| train | 64,620 |
| val | 13,410 |
| test | 15,373 |

추가로 source-disjoint, map-disjoint split도 만들었다. 목적은 단순 episode split보다 더 엄격하게 source/map leakage를 점검하기 위해서다.

수집한 scenario:

| Scenario | Samples | Policy |
|---|---:|---|
| basic | 10,279 | random_walk |
| basic_audio | 10,315 | random_walk |
| basic_notifications | 10,554 | random_walk |
| deadly_corridor | 240 | mixed |
| deathmatch | 6,584 | mixed |
| defend_the_center | 2,594 | mixed |
| defend_the_line | 2,587 | mixed |
| health_gathering | 3,858 | mixed |
| health_gathering_supreme | 2,682 | mixed |
| multi_deathmatch | 10,493 | mixed |
| my_way_home | 9,951 | mixed |
| predict_position | 2,436 | mixed |
| rocket_basic | 10,054 | random_walk |
| simpler_basic | 10,276 | random_walk |
| take_cover | 500 | mixed |

제외된 scenario:

- `cig`
- `cig_with_unknown`
- `multi_duel`

이 세 개는 현재 서버 환경에서 ViZDoom initialization segfault가 발생해 제외했다.

### Horizon dataset

같은 v4 raw data에서 future window만 바꿔 horizon별 dataset도 만들었다.

| Horizon | Future steps | Samples | Train | Val | Test |
|---:|---:|---:|---:|---:|---:|
| 1s | 5 | 93,403 | 64,620 | 13,410 | 15,373 |
| 3s | 15 | 82,848 | 58,317 | 12,647 | 11,884 |
| 5s | 25 | 73,149 | 51,173 | 11,682 | 10,294 |
| 10s | 50 | 52,765 | 36,601 | 8,730 | 7,434 |
| 30s | 150 | 0 | 0 | 0 | 0 |

30초 horizon은 현재 raw episode 길이로는 충분한 future context가 없어 sample이 생성되지 않았다. 즉 30초 예측을 하려면 먼저 더 긴 episode 수집이 필요하다.

---

## 5. 왜 DINOv3 cache를 썼는가

초기에는 `small_cnn`을 사용했지만, 게임 화면의 visual representation을 처음부터 작은 CNN으로 학습하기에는 데이터 규모와 다양성이 부족했다. 그래서 frozen foundation visual encoder를 쓰는 방향으로 전환했다.

최종 주력 방식:

```text
RGB history
→ frozen DINOv3 ConvNeXt-Tiny
→ dense visual token cache
→ downstream path predictor 학습
```

중요한 점은 DINOv3가 경로를 예측하는 것이 아니라는 것이다.

```text
DINOv3 역할:
  RGB frame을 dense visual token으로 바꾼다.

DINOv3가 보지 않는 것:
  ego-motion
  future path
  route ID
  map
  object label
```

v4 cache:

| 항목 | 값 |
|---|---|
| Backbone | DINOv3 ConvNeXt-Tiny |
| Input image size | 256 |
| History frames | 5 |
| Token shape per sample | `[5, 64, 768]` |
| Cached samples | 93,403 |
| Cache size | 약 44GB |

학습 중에는 RGB를 매번 DINO에 넣지 않고, `visual_tokens`를 바로 불러온다.

```text
batch["visual_tokens"] = [B, 5, 64, 768]
```

여기서 `64`는 대략 8×8 screen-space dense spatial tokens이고, `768`은 각 token의 feature dimension이다.

---

## 6. 현재 주력 모델 구조

현재 주력 모델은 `src/models/cue_memory.py`의 `TwoStreamEgocentricCueMemoryPathPredictor` 계열이다.

큰 흐름은 다음과 같다.

```text
cached DINOv3 visual tokens
→ linear projection
→ 2D spatial positional encoding
→ bottleneck adapter
→ spatial relation module
→ TimeSFormer-style temporal/spatial fusion
→ TokenLearner cue selector
→ cue temporal transformer
→ cue memory bank with ego-motion conditioning
→ horizon query decoder
→ constant-velocity prior + learned visual residual
→ future local path [B, H, 2]
```

### 6.1 Input

```text
visual_tokens: [B, T, N, 768]
ego_history:   [B, T, 3]
```

기본값:

```text
T = 5
N = 64
D = 128 after projection
```

### 6.2 Linear projection

DINO feature dimension은 768이지만 downstream model은 hidden dimension 128을 사용한다.

```text
[B, T, N, 768] → [B, T, N, 128]
```

이 projection은 단순 차원 축소만이 아니라, general visual feature를 path prediction task에 맞는 latent space로 옮기는 첫 학습 단계다.

### 6.3 2D spatial positional encoding

DINO token은 64개 token의 feature지만, token index만으로는 화면의 위/아래/왼쪽/오른쪽 정보가 약하다. 그래서 8×8 grid로 보고 sinusoidal 2D positional encoding을 더한다.

```text
tokens = tokens + PE_2D(x, y)
```

같은 “벽처럼 보이는 feature”라도 왼쪽 벽인지, 중앙 통로인지, 오른쪽 모서리인지에 따라 path prediction 의미가 달라지기 때문이다.

### 6.4 Bottleneck adapter

DINO는 frozen이므로, 모든 visual representation을 직접 fine-tuning하지 않는다. 대신 projection 뒤에 작은 adapter를 두어 downstream task에 맞는 보정을 학습한다.

```text
X' = X + Adapter(X)
Adapter: Linear(D→d_bottleneck) → GELU → Dropout → Linear(d_bottleneck→D)
```

이 구조는 전체 backbone을 학습하는 것보다 가볍고, overfitting과 GPU 비용을 줄인다.

### 6.5 Spatial relation module

초기 구현은 QK top-k dynamic spatial graph였다.

```text
Q = Wq X
K = Wk X
V = Wv X
score_ij = q_i^T k_j / sqrt(D)
N(i) = top-k tokens by score_ij
context_i = Σ_j softmax(score_ij) V_j
X_i' = X_i + W_o context_i
```

이것은 level graph나 pathfinding graph가 아니다. 화면 token들 사이의 latent relation mixing이다.

이후 “8×8 token 위에서 graph가 정말 의미 있는가?”라는 비판을 바탕으로 여러 spatial relation variant를 추가했다.

| Variant | 의미 |
|---|---|
| `no_graph` | explicit spatial relation 제거 |
| `topk_graph` | QK top-8 dynamic graph |
| `full_attention` | 64 token 전체 dense attention |
| `local_grid` | 8×8 grid의 고정 local neighbor |
| `relpos_graph` | relative position bias 포함 |
| `contrast_graph` | feature contrast edge 포함 |
| `local_topk_graph` | local prior + top-k |
| `relpos_contrast_local_graph` | relative position + contrast + local prior |

STRNet의 graph module을 그대로 재현한 것은 아니다. STRNet은 feature map 기반 shift/difference message aggregation에 가깝고, 현재 구현은 DINO token 위에서 attention/edge-message 방식으로 adapted한 것이다.

### 6.6 TimeSFormer-style temporal/spatial fusion

사용한 이유는 frame 평균이 아니라 **같은 screen-space token 위치가 시간에 따라 어떻게 변하는지**를 보려는 것이었다.

개념:

```text
per spatial token:
  [frame1 token_i, frame2 token_i, ..., frameT token_i]
  → temporal self-attention

per frame:
  [token1, token2, ..., tokenN]
  → spatial self-attention
```

즉 TimeSFormer의 divided space-time attention 아이디어를 path prediction feature fusion에 맞게 사용했다.

### 6.7 TokenLearner cue selector

64개 token 전체를 계속 들고 가면 계산량도 늘고, 모든 token이 경로 판단에 똑같이 중요하지도 않다. 그래서 TokenLearner-style selector로 frame마다 중요한 cue token `K=8`개를 만든다.

```text
attention_maps = MLP(tokens)              # [B, N, K]
attention = normalize(attention_maps)     # spatial normalization
cue_k = Σ_i attention_{k,i} token_i
```

출력:

```text
[B, T, N, D] → [B, T, K, D]
K = 8
```

현재 paper-aligned variant는 soft attention map 방식이다. hard top-k selection은 별도 ablation으로 다뤘다.

### 6.8 Cue temporal transformer

TokenLearner가 frame별 cue를 만든 뒤에도, 각 cue가 시간에 따라 어떻게 변했는지 다시 봐야 한다.

```text
cues_over_time: [B, T, K, D]
flatten:        [B, T*K, D]
TransformerEncoder
reshape:        [B, T, K, D]
```

이 단계는 “선택된 중요한 cue들의 시간적 변화”를 모델링한다.

### 6.9 Cue memory bank

cue memory bank는 1초 history 안에서 선택된 cue들을 ego-motion과 함께 memory slot에 누적한다.

입력:

```text
cues_over_time: [B, T, K, D]
ego_history:    [B, T, 3]
```

출력:

```text
memory: [B, K, D]
```

attention memory update의 의미:

```text
cue_t + ego_embedding_t
→ memory slot에 content-addressed write
→ gate로 기존 memory를 얼마나 유지할지 결정
```

구체적으로는:

```text
scores = cues @ memory^T
write_weights = softmax(scores)
slot_context = weighted cue summary
gate = sigmoid(...)
candidate = MLP(...)
memory = norm(memory + gate * (candidate - memory))
```

이 memory는 명시적 object memory나 map memory가 아니다. path loss를 통해 학습되는 differentiable cue memory다.

### 6.10 Horizon query decoder

초기 single-vector direct output은 하나의 pooled vector에서 전체 future path를 뽑는 구조였다. 이후 future step마다 learned query를 두는 horizon query decoder로 바꿨다.

```text
learned horizon queries: [H, D]
cross-attention(query, memory)
→ [B, H, D]
→ Linear
→ [B, H, 2]
```

이 구조는 DETR식 learned query decoding에서 가져온 아이디어다. 여기서는 object query가 아니라 future-step query다.

### 6.11 Motion prior + visual residual

최종 출력은 visual branch만으로 직접 만든 path가 아니라, 최근 이동 방향 기반 constant-velocity path에 learned residual을 더한다.

```text
P_cv = constant_velocity_path(ego_history)
ΔP_visual = decoder(memory)
P_pred = P_cv + residual_scale * ΔP_visual
```

이 설계는 학습 안정성을 크게 높였다. 모델은 “기본 이동 관성”을 다시 배울 필요 없이, 시각 단서가 필요한 부분을 보정하면 된다.

---

## 7. 학습 방식

최종 v4 계열 학습 설정:

| 항목 | 설정 |
|---|---|
| model | `cue_memory_path_predictor` |
| visual backbone | `cached_dinov3_convnext_tiny` |
| temporal | TimeSFormer-style |
| cue selector | TokenLearner-style |
| cue tokens | 8 |
| memory | attention memory |
| decoder | horizon query decoder |
| output | deterministic single future path |
| loss | Huber trajectory loss |
| optimizer | AdamW |
| lr | 5e-4 |
| weight decay | 0.001 |
| dropout | 0.2 |
| grad clip | 1.0 |
| mixed precision | enabled |
| balance key | `source_policy` |
| balance mode | sampler + loss weighting |

loss:

```text
L = Huber(P_pred / scale, P_gt / scale)
```

평가 metric:

```text
ADE = mean_t ||P_pred(t) - P_gt(t)||_2
FDE = ||P_pred(H) - P_gt(H)||_2
```

ADE는 전체 future waypoint 평균 오차, FDE는 마지막 waypoint 오차다.

---

## 8. 핵심 결과: horizon별 성능

v4 default scenario dataset에서 single-output DINOv3 TimeSFormer cue-memory model은 모든 trainable horizon에서 constant-velocity baseline을 이겼다.

| Horizon | CV ADE | CV FDE | Model ADE | Model FDE | ADE gain | FDE gain |
|---:|---:|---:|---:|---:|---:|---:|
| 1s | 33.1120 | 51.4413 | 26.8676 | 41.5629 | 18.9% | 19.2% |
| 3s | 75.7201 | 131.6904 | 62.1001 | 103.3531 | 18.0% | 21.5% |
| 5s | 111.2669 | 202.7233 | 88.6020 | 157.0852 | 20.4% | 22.5% |
| 10s | 217.1669 | 408.6508 | 154.5734 | 258.7196 | 28.8% | 36.7% |

해석:

- 짧은 1초에서도 visual cue-memory model이 CV보다 낫다.
- horizon이 길어질수록 visual residual의 상대적 이득이 커진다.
- 특히 10초 FDE에서 개선 폭이 크다.
- 다만 horizon이 길수록 overfitting과 error accumulation 문제가 커져, 더 긴 episode와 regularization이 필요하다.

---

## 9. Baseline 비교

### 9.1 Constant velocity baseline

가장 중요한 내부 baseline이다.

```text
최근 ego-motion 평균 속도를 미래에도 유지한다고 가정
```

이 baseline은 단순하지만 강하다. 특히 짧은 horizon 또는 smooth trajectory에서는 visual model보다 유리할 수 있다.

### 9.2 Paper-adapted proxy baseline

두 paper 방향을 WIT-VZ trajectory task에 맞게 proxy로 변환했다.

| Paper 방향 | WIT-VZ adapter | 주의점 |
|---|---|---|
| Khaleque, Cook, Gow exploratory agent | center-biased exploratory steering / trainable ego-motion motivation baseline | 원 논문의 live motivation state를 그대로 재현한 것은 아님 |
| Xu et al. screen-only navigation/STP-MSTP | last-frame saliency steering / trainable pixels-only DINO baseline | 원 논문의 ARPG STP/MSTP detector를 그대로 재현한 것은 아님 |

Proxy baseline 결과:

| Horizon | Best paper proxy ADE/FDE | Ours ADE/FDE | 해석 |
|---:|---:|---:|---|
| 1s | 36.5431 / 57.1757 | 26.8676 / 41.5629 | ours 우세 |
| 3s | 86.2122 / 153.6574 | 62.1001 / 103.3531 | ours 우세 |
| 5s | 125.0026 / 234.8750 | 88.6020 / 157.0852 | ours 우세 |
| 10s | 198.3879 / 286.8605 | 154.5734 / 258.7196 | ours 우세 |

Trainable paper-inspired 3s baseline:

| Model | ADE | FDE | 해석 |
|---|---:|---:|---|
| Khaleque-inspired ego-motion trainable | 67.9606 | 116.7845 | motion-only learned baseline |
| Xu-inspired pixels-only trainable | 64.4343 | 103.5473 | visual history만으로도 강함 |
| constant velocity | 75.7201 | 131.6904 | 내부 motion prior |
| ours | 62.1001 | 103.3531 | 근소하지만 가장 좋음 |

해석:

- hand-coded proxy보다 trainable proxy가 훨씬 강하다.
- Xu-inspired pixels-only baseline이 강한 것은 screen-only visual signal 자체가 의미 있음을 보여준다.
- 하지만 최종 구조는 ego-motion prior, visual cue, memory, horizon query를 같이 쓰기 때문에 가장 안정적이다.

### 9.3 Privileged oracle baseline

PointNav/DD-PPO와 A*도 baseline으로 만들었지만, 이들은 input-matched competitor가 아니라 **upper-bound**다.

| Baseline | Privilege |
|---|---|
| PointNav/DD-PPO goal-oracle | GT future endpoint를 goal로 받음 |
| A* pose-graph oracle | recorded pose graph와 GT endpoint 사용 |

5초 기준:

| Model | ADE | FDE | 해석 |
|---|---:|---:|---|
| constant velocity | 111.2669 | 202.7233 | recent motion only |
| PointNav oracle | 46.3397 | 0.0000 | GT endpoint privilege |
| A* oracle | 46.9705 | 0.0000 | map/pose/goal privilege |

이 결과는 RGB history만으로 예측하는 모델과 map+goal을 받는 classical planning의 차이를 보여주기 위한 것이다. 직접적인 공정 경쟁으로 말하면 안 된다.

---

## 10. Ablation에서 확인한 것

### 10.1 Cue memory

3초 retraining ablation:

| Variant | Test ADE | Test FDE | 해석 |
|---|---:|---:|---|
| attention memory | 62.9757 | 105.8469 | content-addressed memory |
| attention no ego | 70.8506 | 123.0957 | ego conditioning 제거 시 악화 |
| GRU memory | 62.3833 | 105.8130 | attention memory와 유사 |
| last cue no memory | 70.0242 | 119.2231 | memory update 제거 시 악화 |
| mean cue no memory | 70.8375 | 120.8959 | 단순 평균도 부족 |

해석:

- memory update는 의미가 있다.
- ego-motion conditioning도 중요하다.
- attention memory와 GRU memory는 이 run에서는 비슷했으므로, “attention memory만이 유일한 정답”이라고 말하기보다는 “cue memory update가 필요하다”가 더 안전한 결론이다.

### 10.2 Temporal module

3초 retraining ablation:

| Variant | Test ADE | Test FDE | 해석 |
|---|---:|---:|---|
| TimeSFormer | 61.8506 | 103.5420 | ADE best |
| temporal transformer | 62.9445 | 105.4611 | 비슷하지만 약간 낮음 |
| GRU | 64.7325 | 109.6545 | 상대적으로 낮음 |
| STRNet-style | 62.0693 | 103.8815 | TimeSFormer와 근접 |
| no temporal adapter | 62.1606 | 102.6415 | FDE best |

해석:

- TimeSFormer가 ADE 기준 가장 좋지만, 차이가 아주 크지는 않다.
- 따라서 발표에서는 “시간 변화 모델링은 필요해 보이지만, 현재 데이터에서는 temporal module 종류에 따른 차이는 제한적”이라고 말하는 것이 안전하다.

### 10.3 TokenLearner selector

TokenLearner variant는 horizon별로 비교했다.

| Horizon | Strong variant | 관찰 |
|---:|---|---|
| 1s | query attention / top-k가 TokenLearner보다 약간 좋음 | 차이가 작고 short horizon 영향 제한 |
| 3s | softmax TokenLearner가 FDE 좋음, sigmoid와 근접 | TokenLearner 계열 안정 |
| 5s | sigmoid/softmax/query attention이 근접 | hard top-k는 약간 불리 |

해석:

- TokenLearner의 핵심은 hard top-k가 아니라 soft attention map으로 cue를 압축하는 것이다.
- 현재 결과는 “adaptive cue selection은 유효하지만, 특정 selector 방식 하나가 압도적이라고 보긴 어렵다”에 가깝다.

### 10.4 Spatial graph

처음에는 top-k graph의 의미가 모호했다. “8×8 DINO token 위에서 graph relation이 진짜 필요한가?”라는 비판이 있었고, 이후 full attention, local grid, relative position, contrast-aware edge 등을 추가했다.

10초 모델을 prefix로 잘라 평가한 graph subset 결과:

| Prefix | no graph ADE | top-k graph ADE | best observed ADE | 해석 |
|---:|---:|---:|---:|---|
| 1s | 38.4808 | 32.2773 | 32.2773 | top-k graph 강함 |
| 3s | 70.6455 | 66.1293 | 66.1293 / 66.3595 근접 | top-k와 hybrid 근접 |
| 5s | 101.1153 | 96.6943 | 96.1196 | relpos+contrast+local이 근소 우세 |
| 10s | 167.3324 | 159.7269 | 159.7269 | top-k graph 우세 |

subset에서는 local/relative/contrast가 특정 상황에서 의미가 있었다.

예:

- 3s front-blocked subset: `relpos_contrast_local_graph`가 top-k보다 좋음
- 5s CV-hard subset: `relpos_contrast_local_graph`가 top-k보다 좋음
- 10s left-right asymmetric layout: contrast/local 계열이 top-k보다 좋음

해석:

- 전체 평균에서는 top-k graph가 여전히 강하다.
- 하지만 특정 어려운 subset에서는 relative position, local topology, contrast-aware edge가 보조적으로 의미 있다.
- 이 모듈은 “게임 레벨 그래프”가 아니라 “screen-space visual token relation refinement”로 설명해야 한다.

### 10.5 Long-term memory

현재 기본 모델은 1초 history sample 내부 memory다. 이후 episode chunk 기반 long-term memory도 구현/평가했다.

핵심 질문:

> 5프레임 안의 short memory만으로 충분한가, 아니면 episode-level로 이어지는 장기 cue memory가 필요한가?

CV easy/hard split 기준 결과:

| Horizon | Overall best | Easy best | Hard best | 해석 |
|---:|---|---|---|---|
| 1s | current short window | long mean memory | long gated forget ego | hard subset에서 long memory 유리 |
| 3s | long attention no ego | long gated ego | episodic short only | 3s hard에서는 explicit long memory 불필요 |
| 5s | episodic short only | long attention ego | long gated ego | hard subset에서 long memory 유리 |
| 10s | long gated ego | episodic short only | long attention no ego | hard subset에서 long memory 유리 |

해석:

- long memory가 항상 좋은 것은 아니다.
- 하지만 CV-hard sample, 즉 최근 움직임 외삽이 어려운 sample에서는 long memory가 도움이 되는 경우가 많다.
- 따라서 장기 기억의 주장은 “전체 성능을 항상 올린다”가 아니라 “recent motion만으로 부족한 어려운 subset에서 도움이 된다”로 제한해야 한다.

---

## 11. Human replay GT와 데모 확장

기존 v4 GT는 주로 scripted policy / recorded WIT-VZ trajectory다. “실제 플레이어 움직임과 비교할 수 있나?”라는 질문 때문에 SauerkrautLM public human action labels를 ViZDoom에서 replay하여 human-action replay-derived GT를 만들었다.

주의:

```text
human action label을 ViZDoom에서 replay하여 future path를 만든 것
원본 human pose trajectory를 직접 복원한 것은 아님
```

5초 demo metric:

| Block | Samples | CV ADE/FDE | Ours ADE/FDE | 해석 |
|---|---:|---:|---:|---|
| Human replay GT | 64 | 161.9 / 280.4 | 147.9 / 259.0 | ours가 CV보다 개선 |
| V4 multi-scenario | 9,448 | 119.3 / 217.4 | 93.3 / 164.8 | ours가 CV보다 개선 |

PointNav/A* oracle은 같은 demo에 같이 넣었지만, 이 둘은 GT endpoint를 사용하는 privileged upper-bound다.

---

## 12. 최종 발표 데모 산출물

최종 발표용 영상:

```text
reports/demo/presentation_sequence/demo_final_main_with_external_05s.mp4
```

영상 구성:

1. main real ViZDoom five-baseline counterfactual rollout
2. ViZDoom 5s ADE/FDE result card
3. external dataset sanity section intro
4. MiniWorld / AI2-THOR real CV/GT/Ours simulator rollout
5. external zero-shot 3s metric card
6. ProcTHOR / DeepMind Lab / Habitat / MineDojo overview cards

영상 정보:

| 항목 | 값 |
|---|---:|
| resolution | 2560×1440 |
| FPS | 8 |
| frames | 680 |
| duration | 85.0s |

데모의 핵심 메시지:

- recorded overlay가 아니라, 가능한 곳에서는 simulator branch를 분기해 실제로 다른 1인칭 화면을 렌더링했다.
- CV, PointNav, A*, GT, Ours를 같은 시작 pose에서 비교한다.
- PointNav/A*는 ViZDoom pose-graph setting용 privileged oracle이라 외부 데이터셋 섹션에서는 제외했다.
- 외부 섹션은 broad generalization proof가 아니라 domain-shift sanity check다.

---

## 13. 외부 데이터셋 sanity check

ViZDoom만으로는 “게임 일반화”를 주장하기 어렵다. 그래서 WIT-VZ schema를 다른 simulator/domain에 적용할 수 있는지 확인했다.

확인한 외부 domain:

| Domain | 상태 | 해석 |
|---|---|---|
| MiniWorld | 완료 | schema는 적용되지만 zero-shot 성능은 나쁨 |
| AI2-THOR | 완료 | object-rich Unity scene에서도 pipeline은 동작하지만 checkpoint는 domain shift에 취약 |
| ProcTHOR | 완료 | procedural Unity house에서도 zero-shot 실패가 큼 |
| DeepMind Lab | 완료 | 작은 game-like demo에서는 CV보다 좋아진 유일한 positive sanity case |
| Habitat-Sim | 완료 | photorealistic domain에서 scale/domain shift 큼 |
| MineDojo | overview/gate 성격 | Minecraft-style pose/RGB formulation은 가능하지만 일반화 증거는 아님 |

외부 zero-shot 3s metric:

| Dataset | CV ADE | Ours ADE | Ours FDE | 해석 |
|---|---:|---:|---:|---|
| MiniWorld | 0.250 | 42.156 | 71.734 | strong domain/scale shift |
| AI2-THOR | 1.028 | 51.372 | 83.158 | strong domain/scale shift |
| ProcTHOR | 1.158 | 79.794 | 134.560 | strong domain/scale shift |
| DeepMind Lab | 180.822 | 155.288 | 239.752 | positive small sanity case |
| Habitat | 0.571 | 44.765 | 74.896 | strong domain/scale shift |
| MineDojo | 0.447 | 89.338 | 161.605 | strong domain/scale shift |

해석:

- WIT-VZ input/output formulation은 다른 simulator에도 적용 가능하다.
- 하지만 ViZDoom-trained checkpoint가 broad zero-shot generalization을 한다고 말할 수는 없다.
- DeepMind Lab은 game-like visual/trajectory 구조가 있어 일부 transfer 가능성을 보여주지만, sample 수가 작아 강한 주장으로 쓰면 안 된다.
- 외부 domain에서는 coordinate scale calibration, domain adaptation, external-domain training이 필요하다.

---

## 14. 현재 연구의 기여를 어떻게 말할 수 있는가

가장 안전한 기여 표현은 다음이다.

1. **문제 정의**
   - 1인칭 RGB history와 ego-motion history에서 future egocentric local path를 예측하는 WIT-VZ formulation을 정리했다.

2. **데이터셋 구축**
   - ViZDoom default scenarios 기반으로 multi-scenario path prediction dataset을 구축했다.
   - episode/source/map split과 horizon별 dataset을 만들었다.

3. **모델 구조**
   - frozen DINOv3 dense visual tokens 위에 temporal modeling, adaptive cue selection, cue memory, horizon query decoder, motion residual prior를 결합했다.

4. **정량 비교**
   - 1s/3s/5s/10s horizon에서 constant velocity baseline을 안정적으로 개선했다.
   - paper-adapted proxy baseline과 trainable paper-inspired baseline도 비교했다.
   - privileged oracle baseline을 별도로 제시하여 map/goal privilege와 visual-history prediction의 차이를 분리했다.

5. **모듈 분석**
   - cue memory, temporal module, TokenLearner, spatial graph, long memory를 ablation했다.
   - 특히 memory는 전체 평균보다 CV-hard subset에서 더 의미 있는 것으로 정리했다.

6. **시각화/데모**
   - overlay뿐 아니라 ViZDoom counterfactual rollout demo를 만들었다.
   - 외부 simulator sanity check까지 붙여 formulation portability와 domain shift를 동시에 보여줬다.

---

## 15. 현재 한계

### 15.1 데이터 한계

- v4 dataset은 scripted policy 중심이다.
- human replay GT는 action label replay로 만든 것이며, 원본 human pose trajectory가 아니다.
- 30초 horizon은 현재 raw episode 길이로는 sample이 0개다.
- 일부 scenario는 sample 수가 매우 적다. 특히 `deadly_corridor`, `take_cover`.

### 15.2 모델 한계

- 현재 주력 모델은 deterministic single path predictor다.
- multi-modal route choice가 필요한 상황에서는 하나의 평균 path로 무너질 수 있다.
- graph module은 semantic graph가 아니라 latent token relation이다.
- STRNet을 그대로 재현한 것이 아니라 STRNet-inspired representation component다.
- long memory는 전체 평균에서 항상 좋지 않다.

### 15.3 평가 한계

- PointNav/A*는 privileged oracle이라 공정 baseline이 아니다.
- paper baseline은 exact reproduction이 아니라 WIT-VZ trajectory adapter다.
- 외부 zero-shot 결과는 대부분 domain shift failure다.
- ADE/FDE는 local coordinate scale에 민감하므로 domain 간 절대 수치를 직접 비교하면 안 된다.

### 15.4 일반화 한계

현재 결과로 말할 수 있는 것:

```text
ViZDoom multi-scenario in-domain에서는 visual cue-memory trajectory predictor가 CV보다 낫다.
WIT-VZ formulation은 외부 simulator에도 적용 가능하다.
```

현재 결과로 말하면 안 되는 것:

```text
모델이 게임 전반에 zero-shot generalization한다.
모델이 사람처럼 길찾기를 한다.
STRNet/TokenLearner 논문을 그대로 재현했다.
PointNav/A*보다 공정하게 경쟁했다.
```

---

## 16. 발표 흐름 추천

발표는 다음 순서가 가장 자연스럽다.

1. Problem Motivation
   - 사람은 화면 단서와 시간적 경험으로 다음 경로를 예측한다.

2. Task Definition
   - RGB history + ego-motion → future local path `[forward, right]`

3. Dataset
   - WIT-VZ, ViZDoom v4 default scenarios, split, horizon

4. Model Architecture
   - DINOv3 visual tokens
   - TimeSFormer temporal modeling
   - TokenLearner cue selection
   - Cue memory with ego-motion
   - Horizon query decoder
   - CV prior + visual residual

5. Training/Evaluation
   - Huber loss, ADE/FDE, source-policy balancing

6. Main Results
   - 1s/3s/5s/10s CV vs Ours

7. Baselines
   - paper-adapted proxy
   - trainable paper-inspired
   - PointNav/A* as privileged upper bound

8. Ablations
   - memory, temporal, TokenLearner, graph, long memory
   - emphasis: memory helps especially when CV is insufficient

9. Visualization
   - final counterfactual rollout video
   - human replay block
   - external sanity section

10. Limitations and Next Work
    - scripted data, real human trajectory, longer horizon, domain adaptation, multi-modal future path

---

## 17. 다음 연구 단계

가장 현실적인 다음 단계는 다음 순서다.

1. **데이터 보강**
   - 부족 scenario 보강: `deadly_corridor`, `take_cover`
   - longer episode collection으로 30s horizon 생성
   - 가능한 경우 실제 human play pose trajectory 확보

2. **일반화 실험**
   - 외부 dataset zero-shot이 아니라 external-domain fine-tuning / adapter tuning
   - coordinate scale normalization
   - domain-specific calibration layer

3. **모델 구조 정리**
   - graph module은 top-k graph를 기본으로 두되, subset에서 hybrid graph의 의미를 더 확인
   - long memory는 CV-hard subset 중심으로 주장
   - multi-modal trajectory head는 route ambiguity가 있는 데이터가 확보된 뒤 재검토

4. **평가 강화**
   - seed 반복
   - scenario별 metric
   - CV-hard / turn / obstacle / corridor / asymmetric layout subset report
   - qualitative failure taxonomy

5. **논문화 방향**
   - “게임 navigation generalization”보다 “egocentric visual cue-memory local trajectory prediction”으로 scope를 좁히는 것이 안전하다.
   - novelty는 대형 foundation visual token 위에 cue selection, cue memory, motion residual, horizon query decoding을 결합하고, ViZDoom/external simulator에서 trajectory formulation을 검증한 점에 둔다.

---

## 18. 파일 지도

핵심 코드:

| 파일 | 역할 |
|---|---|
| `src/wit_vz/collect.py` | ViZDoom raw data collection |
| `src/wit_vz/build_samples.py` | raw episode → supervised samples |
| `src/wit_vz/dataset.py` | RGB/cache/ego/path dataset loader |
| `src/models/backbones.py` | small CNN, DINOv2/v3, cached visual token encoder |
| `src/models/cue_memory.py` | main cue-memory path predictor |
| `src/train_path_predictor.py` | training loop |
| `src/eval_path_predictor.py` | checkpoint evaluation |
| `src/metrics.py` | ADE/FDE/per-horizon metrics |
| `src/losses.py` | Huber/multimodal trajectory loss |

핵심 문서/결과:

| 파일 | 내용 |
|---|---|
| `reports/dataset_and_training_method_20260521.md` | v4 dataset/training summary |
| `reports/horizon_sweep_v4_defaults_single_output_20260521.md` | horizon results |
| `reports/paper_baselines_v4.md` | paper-adapted baseline |
| `reports/trainable_paper_baselines_v4_03s.md` | trainable paper-inspired baselines |
| `reports/navigation_oracle_baselines_v4.md` | PointNav/A* oracle baseline |
| `reports/episodic_memory_ablation_v4.md` | long-memory retraining results |
| `reports/cv_easy_hard_episodic_memory_ablation_v4.md` | CV easy/hard subset analysis |
| `reports/graph_subset_ablation_v4_10s.md` | graph subset/prefix ablation |
| `reports/demo_external_generalization_results.md` | external simulator results |
| `reports/demo/presentation_sequence/demo_final_main_with_external_05s.mp4` | final presentation video |

