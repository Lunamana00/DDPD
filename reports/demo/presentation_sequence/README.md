# Presentation Demo Sequence

Use this folder as the shortest live-demo path for slides or screen sharing.

## Full Video

| File | Description |
|---|---|
| `demo_full_sequence.mp4` | 42-second 1080p walkthrough of the full 1-11 demo sequence below. Static contact sheets are held briefly, and GIF segments retain motion. |
| `demo_triptych_moving_sequence.mp4` | 210-second 1080p moving scene walkthrough. Each selected raw episode segment is rendered frame-by-frame with three aligned panels: left = constant-velocity baseline, center = ground truth, right = model prediction. The RGB scene moves over time while each path plot progressively reveals the corresponding trajectory. |
| `demo_paper_baseline_overlay_sequence.mp4` | 210-second 1080p recorded-scene overlay comparison. The same raw RGB future segment is shown with four trajectory panels: CV baseline, Xu-style pixels-only paper baseline, GT, and ours. |
| `demo_navigation_oracle_overlay_05s.mp4` | 25-second 1080p 5-second-horizon overlay comparison. Five panels show CV, PointNav/DD-PPO goal oracle, A* pose-graph oracle, GT, and ours; the video explicitly marks PointNav/A* as privileged upper-bound baselines. |
| `demo_navigation_oracle_counterfactual_rollout_05s.mp4` | 25-second 1080p 5-second-horizon real ViZDoom counterfactual rollout with Korean explanations and explicit baseline names. Five branches are restarted from the same sample pose and followed separately: CV baseline, PointNav/DD-PPO goal-conditioned upper bound, A* classical-planning upper bound, GT, and Ours. |
| `demo_main_5baseline_multiscenario_counterfactual_05s.mp4` | Main 1440p real ViZDoom counterfactual rollout video. It concatenates a human-action replay GT block and a V4 multi-scenario ViZDoom GT block using CV, PointNav/DD-PPO endpoint oracle, A* pose-graph oracle, GT, and ours. Each column is a separate simulator branch from the same selected pose, so the first-person views can diverge. |
| `demo_main_5baseline_multiscenario_05s.mp4` | Recorded-scene overlay version of the same 5-baseline multi-scenario comparison. The RGB future frames are intentionally shared across columns; use it only when the goal is trajectory overlay readability, not as a counterfactual rollout demo. |
| `demo_human_action_replay_gt_comparison_05s.mp4` | 25-second 1080p 5-second-horizon comparison on human-action replay-derived GT. The same replayed ViZDoom future frames are shown with CV, GT, and ours; GT is produced by replaying SauerkrautLM public human action labels, not by recovering original human pose trajectories. |
| `demo_human_action_replay_all_baselines_05s.mp4` | 25-second 1440p 5-second-horizon comparison on human-action replay-derived GT with all selected baselines: CV, Xu-style pixels-only proxy, Khaleque-style exploratory proxy, PointNav/DD-PPO endpoint oracle, A* pose-graph oracle, GT, and ours. PointNav/A* are marked as privileged baselines; GT is produced by replaying SauerkrautLM public human action labels. |
| `demo_vizdoom_counterfactual_rollout.mp4` | Demo-grade ViZDoom counterfactual rollout. Eight 3s ViZDoom samples are restarted from the selected pose, then CV, Xu-style paper baseline, GT, and ours are followed separately to render different first-person videos. |
| `demo_real_counterfactual_rollout_suite.mp4` | 29-second 1080p real counterfactual rollout suite. ViZDoom, MiniWorld, and AI2-THOR are restarted from each selected sample pose, then CV, GT, and ours are followed as separate simulator branches and shown side by side. |
| `demo_real_counterfactual_rollout_suite_cv.mp4` | Branch-only first-person suite for the CV path. |
| `demo_real_counterfactual_rollout_suite_target.mp4` | Branch-only first-person suite for the GT path. |
| `demo_real_counterfactual_rollout_suite_prediction.mp4` | Branch-only first-person suite for the model prediction path. |
| `demo_triptych_sequence.mp4` | 180-second 1080p static sample-by-sample walkthrough. Each scene is rendered as three aligned panels: left = constant-velocity baseline, center = ground truth, right = model prediction. |

## Sequence

| Order | File | Message |
|---:|---|---|
| 1 | `01_vizdoom_3s_overview.png` | In-domain ViZDoom diversity: easy, hard, and failure cases across several scenarios. |
| 2 | `02_vizdoom_hard_my_way_home.gif` | Hard navigation case: constant-velocity extrapolation is poor, visual history helps. |
| 3 | `03_vizdoom_hard_health_gathering.gif` | Object/avoidance case: visual layout and recent motion matter. |
| 4 | `04_vizdoom_failure_predict_position.png` | Failure case: the model can still drift badly in visually noisy or ambiguous samples. |
| 5 | `05_vizdoom_10s_failure_deathmatch.png` | Long-horizon limitation: error compounds as horizon increases. |
| 6 | `06_miniworld_external_overview.png` | External-domain sanity check: the same WIT-VZ formulation runs on MiniWorld, but zero-shot transfer fails strongly. |
| 7 | `07_ai2thor_external_overview.png` | Object-rich Unity-domain check: the same WIT-VZ formulation runs on AI2-THOR, but the ViZDoom checkpoint is not domain-general. |
| 8 | `08_procthor_external_overview.png` | Procedural Unity-house check: source ProcTHOR runs through the same WIT-VZ path, and again exposes zero-shot domain shift. |
| 9 | `09_deepmind_lab_external_overview.png` | Game-like external-domain check: DeepMind Lab runs through the same WIT-VZ path; unlike the simple random-walk external domains, visual prediction improves over CV on this small demo. |
| 10 | `10_habitat_external_overview.png` | Photorealistic embodied-navigation check: Habitat-Sim runs through the same WIT-VZ path, but the ViZDoom checkpoint fails strongly under scale and visual-domain shift. |
| 11 | `11_minedojo_external_overview.png` | Minecraft-style sandbox check: MineDojo exposes RGB and privileged pose, so the WIT-VZ formulation runs, but the zero-shot ViZDoom checkpoint fails badly while CV remains strong on the smooth plains rollout. |

## Claim Boundary

The ViZDoom examples support in-domain scenario diversity, not broad game
generalization. The MiniWorld, AI2-THOR, ProcTHOR, DeepMind Lab, Habitat, and MineDojo examples support
formulation transfer and expose domain shift: the pipeline can run outside
ViZDoom, but the ViZDoom-trained checkpoint is not calibrated for different
dynamics, visual style, or coordinate scale. DeepMind Lab is the current
positive external sanity case, but it is still a small demo rather than a broad
generalization proof. MineDojo should be presented as a formulation/labeling
gate and a domain-gap failure case, not as evidence of Minecraft generalization.
