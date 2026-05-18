# ViZDoom / DoomFrameDataset 구조 정리

이 문서는 `ViZDoom` 기반 게임 데이터셋 중 Hugging Face의 `brahmandam/DoomFrameDataset`을 일부 확인한 결과를 정리한 것이다. 사용자가 말한 `Wizdoom`은 일반적으로 `ViZDoom`으로 표기된다.

- 데이터셋 출처: https://huggingface.co/datasets/brahmandam/DoomFrameDataset
- 로컬 확인 파일: `data/doomframe_sample/train-000000.tar`
- 확인 기준: 첫 번째 WebDataset shard 전체

Route-conditioned architecture에 맞춰 새로 수집하는 데이터셋 구조는 `ROUTE_VIZDOOM_COLLECTION.md`에 별도로 정리했다.

## 1. 데이터셋 성격

`ViZDoom` 자체는 고정된 이미지 데이터셋이라기보다 Doom 게임을 실행하면서 agent가 관측, 행동, 보상을 주고받는 강화학습 환경이다.

`DoomFrameDataset`은 이 환경에서 policy rollout으로 미리 수집된 frame-action 데이터셋이다. 따라서 일반적인 supervised learning, imitation learning, behavior cloning, offline reinforcement learning 실험에 바로 사용할 수 있는 형태다.

기본 단위는 다음과 같다.

```text
image frame + metadata
```

또는 학습 관점에서는 다음처럼 볼 수 있다.

```text
observation: RGB game frame
target/action: action_id, action_name, action_vector
RL metadata: reward, done, value, episode, step
```

## 2. 다운로드한 shard 정보

로컬에서 확인한 shard는 다음 파일이다.

```text
data/doomframe_sample/train-000000.tar
```

확인 결과:

| 항목 | 값 |
| --- | --- |
| 파일 크기 | 2,261,657,600 bytes |
| 포맷 | tar / WebDataset shard |
| PNG 파일 수 | 79,426 |
| JSON 파일 수 | 79,426 |
| 샘플 수 | 79,426 |
| episode 수 | 16 |
| action id 수 | 18 |
| positive reward 샘플 수 | 153 |
| `done: true` 샘플 수 | 15 |
| 이미지 해상도 | 156 x 100 |
| 이미지 포맷 | 24-bit RGB PNG |

전체 데이터셋은 여러 개의 shard로 나뉜 WebDataset 형태이며, 데이터셋 카드 기준 URL 패턴은 다음과 같다.

```text
https://huggingface.co/datasets/brahmandam/DoomFrameDataset/resolve/main/data/train-{000000..000030}.tar
```

## 3. tar 내부 파일 구조

tar 내부는 같은 key를 가진 `.png`와 `.json`이 한 쌍으로 들어 있다.

예:

```text
000000000000.png
000000000000.json
000000000001.png
000000000001.json
000000000002.png
000000000002.json
...
```

즉 `000000000000`이라는 key 하나가 하나의 학습 샘플을 의미한다.

```text
000000000000.png   -> 게임 화면 프레임
000000000000.json  -> 해당 프레임의 행동, 보상, episode 정보
```

로컬에 일부 샘플을 추출해둔 위치:

```text
data/doomframe_sample/first_samples/
```

시각 확인용 contact sheet:

```text
data/doomframe_sample/first_samples/sample_contact_sheet.png
```

## 4. 이미지 데이터

각 `.png` 파일은 agent가 본 Doom 화면이다.

| 항목 | 값 |
| --- | --- |
| 파일 확장자 | `.png` |
| 색상 | RGB |
| 해상도 | 156 x 100 |
| 픽셀 포맷 | 24-bit RGB |
| 평균 파일 크기 | 약 26 KB |

이미지는 모델 입력으로 바로 쓸 수 있다. 일반적인 전처리는 다음 중 하나를 선택한다.

- RGB 그대로 사용
- grayscale 변환
- resize
- pixel value를 `[0, 1]` 또는 `[-1, 1]`로 normalize
- 연속 frame stack 구성

## 5. JSON 메타데이터 구조

샘플 JSON 예시:

```json
{
  "action_id": 1,
  "action_name": "TURN_RIGHT",
  "action_vector": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
  "curriculum_level": 1,
  "done": false,
  "episode": 1,
  "frame_path": "frames/episode_001/step_000000.png",
  "global_step": 0,
  "reward": 0.0,
  "source_frame_path": "frames/episode_001/step_000000.png",
  "step": 0,
  "value": 1.7968196868896484,
  "webdataset_key": "000000000000"
}
```

필드 설명:

| 필드 | 타입 | 의미 |
| --- | --- | --- |
| `action_id` | int | 행동 class id |
| `action_name` | string | 사람이 읽을 수 있는 행동 이름 |
| `action_vector` | list[float] | 멀티 버튼 행동 벡터 |
| `curriculum_level` | int | rollout 수집 시 사용된 curriculum 단계 |
| `done` | bool | episode 종료 여부 |
| `episode` | int | episode 번호 |
| `frame_path` | string | 원래 frame 경로 |
| `global_step` | int | 전체 rollout 기준 step |
| `reward` | float | 해당 step의 reward |
| `source_frame_path` | string | 원본 frame 경로 |
| `step` | int | episode 내부 step |
| `value` | float | rollout policy/value model의 value estimate |
| `webdataset_key` | string | WebDataset 샘플 key |

## 6. Action space

첫 shard에서 확인된 action은 총 18개다.

| action_id | action_name | 첫 shard 내 개수 |
| ---: | --- | ---: |
| 0 | `TURN_LEFT` | 2,272 |
| 1 | `TURN_RIGHT` | 3,148 |
| 2 | `MOVE_RIGHT` | 2,638 |
| 3 | `MOVE_RIGHT+TURN_LEFT` | 4,296 |
| 4 | `MOVE_RIGHT+TURN_RIGHT` | 4,982 |
| 5 | `MOVE_LEFT` | 1,934 |
| 6 | `MOVE_LEFT+TURN_LEFT` | 6,394 |
| 7 | `MOVE_LEFT+TURN_RIGHT` | 5,554 |
| 8 | `MOVE_FORWARD` | 4,338 |
| 9 | `MOVE_FORWARD+TURN_LEFT` | 8,826 |
| 10 | `MOVE_FORWARD+TURN_RIGHT` | 3,422 |
| 11 | `MOVE_FORWARD+MOVE_RIGHT` | 6,124 |
| 12 | `MOVE_FORWARD+MOVE_RIGHT+TURN_LEFT` | 4,084 |
| 13 | `MOVE_FORWARD+MOVE_RIGHT+TURN_RIGHT` | 4,462 |
| 14 | `MOVE_FORWARD+MOVE_LEFT` | 5,108 |
| 15 | `MOVE_FORWARD+MOVE_LEFT+TURN_LEFT` | 6,628 |
| 16 | `MOVE_FORWARD+MOVE_LEFT+TURN_RIGHT` | 2,780 |
| 17 | `ATTACK` | 2,436 |

`action_vector`는 6차원 multi-hot 벡터다. 관측된 샘플 기준으로 각 위치는 다음 버튼에 대응하는 것으로 해석할 수 있다.

```text
[ATTACK, MOVE_FORWARD, MOVE_RIGHT, MOVE_LEFT, TURN_RIGHT, TURN_LEFT]
```

예:

```text
TURN_RIGHT
-> [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]

MOVE_FORWARD+TURN_LEFT
-> [0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

ATTACK
-> [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

## 7. Reward와 episode

첫 shard 기준 reward는 `0.0` 또는 `1.0`으로 나타났다.

| 항목 | 값 |
| --- | --- |
| reward 최솟값 | 0.0 |
| reward 최댓값 | 1.0 |
| reward 평균 | 약 0.00193 |
| positive reward 개수 | 153 |

첫 positive reward 샘플:

```json
{
  "webdataset_key": "000000001107",
  "episode": 1,
  "step": 1107,
  "action_name": "ATTACK",
  "reward": 1.0,
  "done": false
}
```

첫 episode 종료 샘플:

```json
{
  "webdataset_key": "000000004997",
  "episode": 1,
  "step": 4997,
  "action_name": "MOVE_FORWARD",
  "reward": 0.0,
  "done": true
}
```

첫 shard에는 episode `1`부터 `16`까지 들어 있다. episode 길이는 최소 `3,736`, 최대 `5,106`, 평균 약 `4,964` step이었다.

## 8. 학습 태스크별 사용 방법

### Behavior cloning / imitation learning

가장 직접적인 사용 방식이다.

```text
input  = PNG frame
target = action_id
```

또는 multi-label 버튼 예측으로 보면 다음처럼 쓸 수 있다.

```text
input  = PNG frame
target = action_vector
```

### Offline reinforcement learning

각 샘플에는 `reward`, `done`, `episode`, `step`이 있으므로 episode 내부에서 다음 step을 이어 붙여 transition을 만들 수 있다.

```text
s_t      = current PNG
a_t      = action_id or action_vector
r_t      = reward
s_{t+1}  = same episode의 다음 step PNG
done_t   = done
```

다만 JSON에 `next_frame_path`가 명시적으로 들어 있지는 않다. 따라서 `episode`와 `step` 순서를 기준으로 직접 연결해야 한다.

### Value prediction

`value` 필드를 target으로 사용하면 policy rollout 당시의 value estimate를 회귀 대상으로 삼을 수 있다.

```text
input  = PNG frame
target = value
```

## 9. Python에서 읽는 예시

tar를 직접 읽는 최소 예시:

```python
import json
import tarfile
from pathlib import Path

shard_path = Path("data/doomframe_sample/train-000000.tar")

with tarfile.open(shard_path, "r") as tar:
    png_member = tar.getmember("000000000000.png")
    json_member = tar.getmember("000000000000.json")

    png_bytes = tar.extractfile(png_member).read()
    metadata = json.load(tar.extractfile(json_member))

print(len(png_bytes))
print(metadata["action_name"], metadata["reward"], metadata["done"])
```

WebDataset 라이브러리를 쓰면 shard streaming 방식으로 학습 파이프라인을 구성할 수 있다.

```python
import webdataset as wds

dataset = (
    wds.WebDataset("data/doomframe_sample/train-000000.tar")
    .decode("pil")
    .to_tuple("png", "json")
)

for image, metadata in dataset:
    action_id = metadata["action_id"]
    reward = metadata["reward"]
    break
```

## 10. 주의할 점

- 이 데이터는 ViZDoom simulator가 아니라 simulator에서 미리 수집한 rollout dataset이다.
- 첫 shard만 확인했기 때문에 전체 데이터셋의 episode/action/reward 분포는 달라질 수 있다.
- `action_vector` 차원 이름은 JSON에 별도 필드로 들어 있지 않으므로 `action_name`과 벡터 패턴을 함께 확인해야 한다.
- offline RL transition을 만들 때 episode 경계를 반드시 확인해야 한다. `done: true` 이후의 다음 frame을 같은 transition으로 연결하면 안 된다.
- reward가 매우 sparse하다. 첫 shard 기준 positive reward 비율은 약 0.19%다.
