# Worked Example: Qwen3.5-4B

This repository is the reference implementation for the skill.

## Real architecture facts

- Dense Qwen3.5-4B text decoder
- Hidden width: 2,560
- Layers: 32
- Schedule: `8 x [3 Gated DeltaNet + 1 gated full attention]`
- DeltaNet: 16 Q/K heads, 32 value heads, head width 128
- Full attention: 16 Q heads, 4 K/V heads, head width 256
- Dense SwiGLU width: 9,216
- Vocabulary: 248,320
- Tied embedding and LM head

## Teaching configuration

- Prompt: `A tiny robot learns`
- Sequence length: 4
- Hidden width: 8
- One detailed Gated DeltaNet layer
- Two numerically executed but collapsed repeated DeltaNet layers
- One detailed full-attention layer
- DeltaNet: 1 Q/K head, 2 value heads, width 2
- Attention: 4 Q heads, 1 K/V head, width 2
- FFN width: 12
- Vocabulary: 8

## Important design decisions

### Recurrent state

The tutorial first shows:

```text
S_-1 -> S_0 -> S_1 -> S_2 -> S_3
```

Then it serializes:

1. Decay
2. Predict `kS`
3. Correct `beta(v - memory)`
4. Write `k^T delta`
5. Update state
6. Read `qS`
7. Stack reads and concatenate heads into `O_core`

Token buttons are inspection checkpoints, not independent states.

### Attention ratio

The original mini design used 2 Q heads and 1 K/V head, which did not preserve the real 4:1
grouping. It was corrected to 4 Q heads and 1 K/V head.

### Repeated layers

Only layer 0 Gated DeltaNet is explained in detail. Layers 1 and 2 still run the complete numerical
forward pass, but one fast-forward step shows:

```text
X_after_L0 -> X_after_L1 -> X_after_L2
```

Each repeated layer has different weights and separate recurrent states.

### Output target

Vocabulary row 5 (`fast`) is a fixed tied-embedding row. Because `fast` is absent from the input,
this changes only the LM-head comparison and leaves all prompt embeddings and hidden activations
unchanged. The tutorial discloses that `fast` is an intentionally designed toy target.

## Explanation format

Every step contains:

- Base narrative
- Why the operation exists
- Exact input provenance
- Exact output handoff
- Forward equation
- Numbered cell computation

The attention O projection, for example, explains:

- `C_gated` was assembled from four attention contexts and saved gates
- `W_o` mixes channels from all heads
- Token mixing already happened in `QK^T`, softmax, and `PV`
- Mini shape is 8 x 8
- Real shape is 4,096 x 2,560
- Result `M` is added to the residual stream next

## Validation results

The reference implementation verifies:

- All visualized matmul cells
- Head-sharing ratios
- Recurrent phase order
- Softmax row sums
- Deep links
- Desktop and mobile browser rendering
- GitHub Pages deployment
