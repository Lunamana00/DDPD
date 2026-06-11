# Presentation Demo Sequence

Use this folder as the shortest live-demo path for slides or screen sharing.

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

## Claim Boundary

The ViZDoom examples support in-domain scenario diversity, not broad game
generalization. The MiniWorld, AI2-THOR, and ProcTHOR examples support
formulation transfer and expose domain shift: the pipeline can run outside
ViZDoom, but the ViZDoom-trained checkpoint is not calibrated for different
dynamics, visual style, or coordinate scale.
