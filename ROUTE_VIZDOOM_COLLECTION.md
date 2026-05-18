# Route-Conditioned ViZDoom Dataset Collection

이 문서는 첨부한 `Route-Conditioned Cue-Memory Transformer` 아키텍처에 맞는 ViZDoom rollout 데이터셋을 새로 수집하는 방법을 정리한다.

기존 `DoomFrameDataset`은 RGB frame과 action은 있지만 route 후보, player pose, chosen route label이 없었다. 이 수집기는 ViZDoom을 직접 실행하면서 다음 정보를 함께 저장한다.

- egocentric RGB frame
- depth buffer
- labels buffer
- automap buffer
- visible object labels
- player pose: `POSITION_X`, `POSITION_Y`, `POSITION_Z`, `ANGLE`
- action: `action_id`, `action_name`, `action_vector`
- reward / done
- candidate route ids
- route distance / progress
- nearest route
- episode-level chosen route

## Added Files

```text
scripts/collect_route_vizdoom.py
configs/vizdoom_route_specs/deadly_corridor_three_lanes.json
requirements-vizdoom.txt
```

## Install

```powershell
uv pip install -r requirements-vizdoom.txt
```

현재 로컬 venv에는 다음 패키지 설치와 smoke test를 완료했다.

```text
vizdoom==1.3.0
pillow==12.2.0
```

## Quick Run

짧은 테스트:

```powershell
uv run python scripts\collect_route_vizdoom.py `
  --episodes 1 `
  --max-steps 20 `
  --run-name smoke_test `
  --overwrite `
  --policy random
```

조금 더 긴 demo 수집:

```powershell
uv run python scripts\collect_route_vizdoom.py `
  --episodes 3 `
  --max-steps 300 `
  --run-name route_demo_3ep `
  --overwrite `
  --policy forward_bias
```

생성 위치:

```text
data/route_vizdoom/runs/<run_name>/
```

로컬에서 생성된 demo:

```text
data/route_vizdoom/runs/route_demo_3ep/
```

## Output Layout

```text
data/route_vizdoom/runs/route_demo_3ep/
  manifest.json
  route_spec.json
  episodes/
    episode_000001/
      summary.json
      steps.jsonl
      frames/
        000000.png
        000001.png
      depth/
        000000.npz
        000001.npz
      labels/
        000000.npz
        000001.npz
      automap/
        000000.png
        000001.png
```

## Manifest

`manifest.json`은 run 전체의 schema와 episode 요약을 담는다.

예시:

```json
{
  "source": "ViZDoom rollout",
  "scenario": "deadly_corridor",
  "map": "map01",
  "num_episodes": 3,
  "num_steps": 106,
  "button_order": [
    "ATTACK",
    "MOVE_FORWARD",
    "MOVE_RIGHT",
    "MOVE_LEFT",
    "TURN_RIGHT",
    "TURN_LEFT"
  ],
  "observation_files": {
    "rgb": "episodes/<episode_id>/frames/<step>.png",
    "depth": "episodes/<episode_id>/depth/<step>.npz",
    "labels": "episodes/<episode_id>/labels/<step>.npz",
    "automap": "episodes/<episode_id>/automap/<step>.png",
    "metadata": "episodes/<episode_id>/steps.jsonl"
  }
}
```

## Step Record Schema

각 episode의 `steps.jsonl`은 step당 한 줄의 JSON이다.

주요 필드:

| Field | Meaning |
| --- | --- |
| `sample_id` | episode와 step을 합친 샘플 id |
| `frame_path` | RGB frame path |
| `depth_path` | depth buffer `.npz` path |
| `labels_path` | labels buffer `.npz` path |
| `automap_path` | automap image path |
| `pose` | player `x`, `y`, `z`, `angle` |
| `game_variables` | ViZDoom game variables |
| `visible_labels` | 현재 frame에 보이는 object labels |
| `candidate_route_ids` | route 후보 id 목록 |
| `route_metrics` | route별 distance/progress |
| `nearest_route_id` | 현재 위치에서 가장 가까운 route |
| `chosen_route_id` | episode 전체 trajectory 기준 route label |
| `action` | action id/name/vector |
| `reward` | step reward |
| `done` | episode 종료 여부 |

예시:

```json
{
  "sample_id": "episode_000001_000000",
  "frame_path": "episodes/episode_000001/frames/000000.png",
  "pose": {
    "x": 0.0,
    "y": 0.0,
    "z": 0.0,
    "angle": 0.0
  },
  "candidate_route_ids": [
    "left_lane",
    "center_lane",
    "right_lane"
  ],
  "nearest_route_id": "center_lane",
  "chosen_route_id": "center_lane",
  "action": {
    "action_id": 10,
    "action_name": "MOVE_FORWARD+TURN_RIGHT",
    "action_vector": [0, 1, 0, 0, 1, 0]
  },
  "reward": 7.0961456298828125,
  "done": false
}
```

## Mapping to the Architecture

| Architecture Part | Collected Field |
| --- | --- |
| Egocentric visual history `I_1...I_T` | `frames/*.png` |
| Auxiliary metadata | `depth/*.npz`, `labels/*.npz`, `automap/*.png`, `pose`, `visible_labels`, `action` |
| Candidate routes `R_1...R_K` | `route_spec.json` and `candidate_route_ids` |
| Route-conditioned cross-attention inputs | `route_metrics`, route waypoints, frame history |
| Supervision: chosen route `y` | `chosen_route_id` |
| Empirical distribution `q` | aggregate `chosen_route_id` over repeated rollouts from the same route-spec/map condition |
| Belief over time | train on prefixes of `steps.jsonl` and predict final `chosen_route_id` |
| Route readability / deceptive-risk critic | compare predicted route distribution with designer target distribution |

## Route Spec

Route candidates are defined in:

```text
configs/vizdoom_route_specs/deadly_corridor_three_lanes.json
```

Each route is a polyline in ViZDoom map coordinates.

```json
{
  "id": "center_lane",
  "waypoints": [
    [-384.0, 0.0],
    [-128.0, 0.0],
    [128.0, 0.0],
    [384.0, 0.0]
  ]
}
```

The collector computes distance and progress from the player pose to each route polyline. The episode-level route label is assigned by majority vote over nearest routes after `ignore_first_steps`.

## Important Limitation

The included `deadly_corridor_three_lanes.json` is a bootstrap route spec. It is useful for validating the data format and end-to-end collection, but route labels should not be treated as final human route-choice ground truth until route waypoints are calibrated for the exact map/design being studied.

For the actual architecture experiment, the recommended process is:

1. Choose or create a ViZDoom map with explicit route alternatives.
2. Define route waypoints in `configs/vizdoom_route_specs/*.json`.
3. Run multiple rollout policies or human-controlled sessions per map condition.
4. Aggregate `chosen_route_id` labels into empirical route distributions.
5. Train the model on visual-history prefixes to predict route belief over time.
