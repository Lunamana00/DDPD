# Model Architecture Paper Alignment

This note maps the current cue-memory path predictor modules to the closest papers and records which parts are implemented.

## Cue Token Selector

Closest paper: TokenLearner, "Adaptive Space-Time Tokenization for Videos" (Ryoo et al., NeurIPS 2021).

Implemented path:

- `selector_type=tokenlearner`
- learns `S` spatial attention maps from frame-local visual tokens
- normalizes each map over spatial tokens
- produces `S` cue tokens by weighted summation over visual tokens
- refines the cue tokens with cue-token self-attention and an MLP

This matches TokenLearner's core tokenizer mechanism. It does not implement the optional TokenFuser remapping stage because the path predictor consumes cue tokens directly rather than reconstructing a dense visual feature map.

Legacy paths:

- `selector_type=query_attention`: DETR/Perceiver-style learned latent query pooling
- `selector_type=topk_tokenlearner`: hard score-gated Top-K selection, useful for ablation but less faithful to TokenLearner

Reference: https://proceedings.neurips.cc/paper/2021/hash/6a30e32e56fce5cf381895dfe6ca7b6f-Abstract.html

## Temporal Adapter

Closest papers: TimeSformer, "Is Space-Time Attention All You Need for Video Understanding?" (Bertasius et al., 2021), and STRNet, "Visual Navigation with Spatio-Temporal Representation through Dynamic Graph Aggregation" (Ren et al., 2026).

Implemented path:

- `temporal_type=timesformer`
- applies temporal attention per spatial token
- applies spatial attention per frame
- keeps temporal positional encoding
- optionally applies temporal channel shift with `use_temporal_shift=true`
- optionally applies multi-resolution temporal difference mixing with `use_temporal_difference_conv=true`
- optionally applies dynamic frame-local spatial graph aggregation before temporal modeling with `use_spatial_graph=true`

This matches the divided temporal/spatial attention idea from TimeSformer and implements STRNet-inspired spatial graph reasoning plus temporal shift/difference mixing. It is not a full STRNet reproduction because the project has no goal-observation stream and no navigation policy head.

References:

- https://arxiv.org/abs/2102.05095
- https://arxiv.org/abs/2604.02829

## Cue Memory Bank

Closest paper: Memory Networks (Weston et al., 2014), plus standard recurrent memory updates.

Implemented path:

- `memory_type=attention`
- keeps a fixed set of learned memory slots
- projects current cue tokens with ego-motion context
- performs content-addressed writes by attention from cue tokens to memory slots
- updates each memory slot with a learned gate and candidate state
- exposes the memory slots to the horizon query decoder for cross-attention reads

This captures the core read/write memory-network idea in a compact differentiable module. It is not a QA-style Memory Network with explicit symbolic facts or multi-hop answer scoring; it is adapted to dense visual cue memory for trajectory prediction.

Legacy path:

- `memory_type=gru_cell`: per-slot recurrent GRUCell update

Reference: https://arxiv.org/abs/1410.3916

## Path Output Head

Implemented path:

- deterministic horizon-query decoder
- outputs one future local trajectory shaped `[B, H, 2]`
- predicts a residual over a constant-velocity motion prior
- uses ADE/FDE in egocentric local coordinates for evaluation

This is the paper-facing main model. It should be described as a single-output
future path predictor.
