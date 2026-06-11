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
