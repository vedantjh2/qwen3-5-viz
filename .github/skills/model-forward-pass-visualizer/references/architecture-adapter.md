# Architecture Adapter Guide

Use this guide to translate an official model implementation into a small but faithful teaching
model.

## Source hierarchy

Prefer sources in this order:

1. Released checkpoint configuration
2. Official framework implementation for that exact model version
3. Official model card
4. Official paper or technical report
5. Maintainer documentation
6. High-quality secondary explanations only for intuition

When sources disagree, trust the executable implementation for operation order and tensor shapes,
then document the disagreement.

## Architecture inventory template

Record this table before implementation:

| Stage | Input shape | Learned values | Computation | Output shape | State/cache |
|---|---|---|---|---|---|
| Embedding | | | | | |
| Pre-mixer norm | | | | | |
| Token mixer projections | | | | | |
| Token mixer core | | | | | |
| Mixer output projection | | | | | |
| First residual | | | | | |
| Pre-FFN norm | | | | | |
| FFN/router | | | | | |
| Second residual | | | | | |
| Final norm | | | | | |
| Output head | | | | | |

Also record:

- Layer schedule
- Head counts and sharing groups
- Head dimensions
- Rotary dimensions and base
- Attention mask type
- Sliding-window or global pattern
- Convolution kernel and grouping
- Recurrent state axes
- Cache axes
- Activation functions
- Bias presence
- Weight tying

## Miniaturization rules

Preserve ratios before absolute sizes.

### Multi-head attention

For real `Hq` query heads and `Hkv` K/V heads:

```text
group_ratio = Hq / Hkv
```

Choose mini counts with the same integer ratio. For example:

```text
16 Q : 4 K/V = 4:1
mini -> 4 Q : 1 K/V
```

Keep concatenated query/context width compatible with the mini residual width when practical.

### Partial RoPE

Preserve the existence and placement of partial rotation. If the exact fraction cannot be shown with
an even two-dimensional rotary pair, state the teaching deviation. Never claim full-head rotation
when the real model rotates only a subset.

### Gated DeltaNet or linear recurrent attention

Preserve:

- Q/K-to-V head ratio
- Key and value head dimensions
- Causal convolution placement
- Q/K normalization
- Query scaling
- beta/write-rate transform
- decay transform
- Per-value-head state shape
- Predict/correct/write/update/read ordering
- Output gate and projection

Show one state chain per value head, not one independently initialized state per token.

### Classic SSMs

Identify whether the implementation uses selective scan, discretized A/B/C parameters, convolution,
gating, and residual branches. Show the state update in the exact implementation convention.

Do not label every recurrent linear-attention model as Mamba.

### MoE

Preserve:

- Router logits
- Router normalization
- Top-k selection
- Capacity or token-dropping behavior
- Expert input/output shapes
- Expert aggregation weights
- Shared experts

Teach routing once for a small token set, expand one selected expert's matmuls, and summarize
parallel experts without duplicating identical FFNs.

### Multimodal models

Separate modality encoders from the language-decoder path. Show:

- Patch or feature construction
- Modality projection into decoder width
- Special-token insertion or replacement
- Position-ID construction
- The point where modalities share the residual stream

If the request is text-only, omit the vision/audio encoder but explain where its projected tokens
would enter.

## Repeated layers

Choose one representative block of each distinct type:

- One recurrent/SSM block
- One local-attention block
- One global-attention block
- One MoE block
- One cross-attention block

Explain each distinct graph fully once. Execute repeated copies numerically and show a compact
fast-forward with:

- Input matrix
- Intermediate layer outputs
- New parameter/state labels
- Final matrix feeding the next distinct block

## State and cache questions

For every stateful operator, answer:

1. What are the state axes?
2. Is state independent per layer?
3. Is state independent per head?
4. Is state independent per batch item?
5. Does state grow with sequence length?
6. Is state reset between prompts?
7. How does prefill differ from single-token decode?
8. What exact tensor is cached?

Include these answers in the tutorial before exposing token checkpoint controls.
