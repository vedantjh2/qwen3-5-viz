# Qwen3.5 Forward-Pass Visualization Guide

## Purpose

This repository is a fork of the original `llm-viz` project. The new `/qwen` experience explains one
four-layer slice of the dense Qwen3.5 text decoder:

1. Gated DeltaNet layer
2. Gated DeltaNet layer
3. Gated DeltaNet layer
4. Gated full-attention layer

Qwen3.5-4B repeats that quartet eight times for 32 decoder layers. The visualization expands one
quartet with small deterministic tensors so every computed matrix cell can be inspected.

The nested `llm-viz/` directory is a read-only reference copy. Product code for this fork lives in the
base repository.

## Architectural ground truth

The implementation and dimensions are based on:

- Qwen3.5-4B model card:
  `https://huggingface.co/Qwen/Qwen3.5-4B`
- Qwen3.5-4B configuration:
  `https://huggingface.co/Qwen/Qwen3.5-4B/raw/main/config.json`
- Hugging Face Qwen3.5 model definition:
  `https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modular_qwen3_5.py`
- The inherited Gated DeltaNet and gated-attention implementation:
  `https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modular_qwen3_next.py`

Qwen calls the first three mixers `linear_attention` layers implemented by `Qwen3_5GatedDeltaNet`.
The UI also calls them "SSM-like recurrent mixers" to connect with the common SSM mental model, but
they are specifically Gated DeltaNet layers rather than a classic Mamba-style state-space layer.

## Real Qwen3.5-4B dimensions

| Quantity | Qwen3.5-4B |
|---|---:|
| Vocabulary | 248,320 |
| Hidden width | 2,560 |
| Decoder layers | 32 |
| Layer layout | `8 x [3 DeltaNet + 1 full attention]` |
| DeltaNet Q/K heads | 16 |
| DeltaNet value heads | 32 |
| DeltaNet Q/K and value head width | 128 |
| DeltaNet causal-convolution kernel | 4 |
| Full-attention query heads | 16 |
| Full-attention K/V heads | 4 |
| Full-attention head width | 256 |
| Rotary dimensions per attention head | 64 |
| Dense FFN intermediate width | 9,216 |
| Native context | 262,144 |
| Token embedding / LM head | tied |

The 4B checkpoint is dense. It has no mixture-of-experts router or expert dispatch in the text
decoder.

## Teaching-model dimensions

The visualization uses a deliberately small "Qwen3.5 Mini" arithmetic model. It is not an official
checkpoint and does not use pretrained Qwen weights.

| Quantity | Teaching value | Preserved relationship |
|---|---:|---|
| Tokens | 4 | causal sequence |
| Vocabulary | 8 | embedding lookup and tied output head |
| Hidden width | 8 | residual stream |
| Decoder layers | 4 | one complete 3:1 quartet |
| DeltaNet Q/K heads | 1 | shared by two value heads |
| DeltaNet value heads | 2 | real model also has twice as many V heads as Q/K heads |
| DeltaNet head width | 2 | fixed key-by-value state remains visible |
| Attention query heads | 4 | preserves the real 4:1 grouped-query ratio |
| Attention K/V heads | 1 | one K/V head shared by four Q heads |
| Attention head width | 2 | keeps concatenated Q/context width at 8 |
| FFN width | 12 | compressed SwiGLU expansion |
| Convolution kernel | 4 | exact real kernel length |

All toy weights are deterministically generated. They exist only to create stable arithmetic for the
tutorial. Do not describe the toy next-token result as a meaningful model prediction.

Vocabulary row 5 (`fast`) is a fixed teaching constant rather than a seeded row. `fast` is absent
from the prompt, so this changes neither token embeddings nor any decoder hidden state; it only makes
the tied LM-head comparison choose the intuitive continuation `A tiny robot learns fast`. Describe
this as an intentionally designed toy target, not learned language behavior.

The teaching attention therefore uses `4 Q heads : 1 K/V head`, exactly matching the grouping ratio
of the real model's `16 Q heads : 4 K/V heads`. Each mini head is compressed to one visible
two-dimensional rotary pair; this keeps the diagrams legible while preserving the head-sharing
topology.

## End-to-end text forward pass

### 1. Token embedding

The tokenizer produces integer IDs. Embedding is a row lookup, not a matrix multiplication:

```text
X[t, :] = E[token_id[t], :]
```

Qwen3.5 does not add an absolute position-embedding matrix to `X`. Position enters the full-attention
layers through partial RoPE. DeltaNet gets order information from its causal convolution and
left-to-right recurrent scan.

### 2. Decoder-layer shell

Every DeltaNet and full-attention layer uses the same pre-norm, two-residual skeleton:

```text
R0 = X
N0 = RMSNorm(X)
M  = TokenMixer(N0)
X1 = R0 + M

R1 = X1
N1 = RMSNorm(X1)
F  = DenseSwiGLU(N1)
X2 = R1 + F
```

`TokenMixer` is Gated DeltaNet in layers 0-2 of each quartet and gated grouped-query attention in
layer 3.

### 3. Gated DeltaNet mixer

For normalized input `N`:

```text
[Q_raw | K_raw | V_raw] = N W_qkv
z                        = N W_z
b                        = N W_b
a                        = N W_a
beta                     = sigmoid(b)
g                        = -exp(A_log) * softplus(a + dt_bias)
```

`Q_raw`, `K_raw`, and `V_raw` pass through a four-tap depthwise causal convolution followed by SiLU.
Depthwise means each projected channel has its own kernel and does not mix with other channels.

After splitting into heads, Q and K are L2-normalized. Q is also scaled by
`1 / sqrt(key_head_dim)`. Q/K heads are repeated as needed to match the larger number of value heads.

The tutorial displays the exact token-by-token recurrent rule used for single-token decode:

```text
S_tilde = exp(g_t) * S_(t-1)
memory  = k_t S_tilde
delta   = beta_t * (v_t - memory)
write   = k_t^T delta
S_t     = S_tilde + write
o_t     = q_t S_t
```

`S` is a fixed `key_head_dim x value_head_dim` matrix per value head. It does not grow with sequence
length. Production prefill normally evaluates an algebraically equivalent chunked delta rule for
throughput; the recurrent form is shown because each read and write can be traced directly.

There is one evolving `S` per value head, not an independently initialized state per token:

```text
S_-1 --token 0--> S_0 --token 1--> S_1 --token 2--> S_2 --token 3--> S_3
```

Each token produces its own `o_t`, but it reads and updates the state inherited from the previous
token. After the scan, `o_0 ... o_(T-1)` are stacked down the time axis for each head, and the value
heads are concatenated across the feature axis to form `O_core`.

The head outputs then use gated RMSNorm and an output projection:

```text
O_gated = RMSNorm(O_core) * SiLU(z)
M       = O_gated W_out
```

### 4. Gated grouped-query full attention

For normalized input `N`:

```text
[Q | output_gate] = N W_qg
K                 = N W_k
V                 = N W_v
```

Q and K receive per-head RMSNorm. Partial RoPE rotates 64 of 256 dimensions in the real model. V is
not rotated. Four K/V heads are shared across 16 query heads.

For each query head:

```text
Scores  = Q K^T / sqrt(head_dim)
Masked  = causal_mask(Scores)
P       = softmax_rows(Masked)
Context = P V
```

The concatenated head output is gated before the output projection:

```text
Context_gated = concat(Context_heads) * sigmoid(output_gate)
M             = Context_gated W_o
```

Unlike DeltaNet, full attention explicitly materializes a token-by-token `T x T` score/probability
matrix.

### 5. Dense SwiGLU FFN

Every decoder layer, including all four layers in the quartet, uses the same dense feed-forward
network:

```text
G = N1 W_gate
P = N1 W_up
H = SiLU(G) * P
F = H W_down
```

There is no expert routing in Qwen3.5-4B.

### 6. Final norm and tied LM head

After all decoder layers:

```text
H_final = RMSNorm(X_final)
Logits  = H_final E^T
P_vocab = softmax(Logits[last_token, :])
```

The 4B model ties `E` between input embedding and output vocabulary projection.

## Visualization contract

- **Blue always means learned/pretrained values.** This includes embeddings, projection matrices,
  normalization scales, convolution kernels, `A_log`, and `dt_bias`.
- **Green always means computed values.** This includes activations, recurrent states, attention
  probabilities, residual streams, logits, and output cells.
- The continuous vertical green arrow is the forward/residual data path.
- Blue side arrows feed learned matrices into projection or normalization operations.
- In every matrix multiplication, click a green result cell to select it.
- The active input-row element and weight-column element are highlighted term by term.
- The running sum must equal the selected result cell. If an operation has an explicit scale such as
  `1 / sqrt(head_dim)`, that scale is applied to each displayed term.
- Non-matmul operations still expose the selected cell's exact formula.

Do not invert or weaken the blue/green meaning. It is shared with the original nano-GPT
visualization and is a core teaching cue.

## Tutorial structure

The sidebar contains 65 ordered steps in five chapters:

1. Orientation
2. Layer 0: Gated DeltaNet
3. Layers 1-2: fast-forward repeated Gated DeltaNet
4. Layer 3: Gated full attention
5. Output head

The architecture rail stays top-down and automatically centers the active operation. Sidebar chapter
controls, arrow keys, and replay controls all update the same active-step state.

Every sidebar step has four explanation layers:

1. The operation-specific narrative
2. A longer "Why this operation exists" section covering shape, modeling purpose, and what the
   operation does or does not mix
3. A "Continuity and provenance" section naming the exact earlier step that produced each reused
   matrix, while distinguishing blue learned parameters from green forward-pass results
4. A numbered cell-level execution breakdown plus the destination that consumes the result next

When a value is deliberately saved and reused much later, such as DeltaNet `z`, attention output
gates, FFN normalized input `U`, shared `V0`, or the tied embedding table, always name its original
calculation step explicitly.

The canonical learning path must not hide dependent work behind tabs:

- Each matrix multiplication is its own Continue step.
- Parallel branches are announced explicitly, then serialized for explanation. For example, the FFN
  gate and up projections can run in parallel in an implementation, but the tutorial explains gate
  first and up second.
- The first DeltaNet layer is explained in full, including state-chain overview, decay, predict,
  correct, write, update, read, and explicit `o_t`-to-`O_core` assembly.
- Layers 1 and 2 are still numerically executed, but their repeated tutorial content is collapsed
  into one fast-forward step. They use the same graph with distinct learned weights and distinct
  per-layer recurrent states.
- Causal masking and row softmax are separate steps.
- Query/key/value projections, RoPE head transforms, attention-score heads, and probability-times-value
  heads are each visited one at a time.
- A cell animation plays from the first inner-dimension term to the last exactly once, then stops.
  Replay restarts that one calculation; it never loops indefinitely or advances the chapter
  automatically.

## Relationship to the original `llm-viz` walkthrough

The Qwen port intentionally follows the original choreography rather than presenting a disconnected
dashboard:

| Original nano-GPT teaching unit | Qwen3.5 replacement |
|---|---|
| Token and position embedding | Token lookup only; Qwen position enters through partial RoPE in full-attention layers |
| LayerNorm walkthrough | RMSNorm before each mixer and FFN, plus final RMSNorm |
| Q/K/V projection | Packed DeltaNet QKV projection or gated-attention Q/gate, K, and V projections |
| Self-attention dot products | Three DeltaNet state scans, then one full-attention QK-transpose and probability-times-V pair |
| GELU MLP | Dense SwiGLU gate/up branches and down projection |
| Projection and residual | DeltaNet `W_out` or attention `W_o`, followed by the same residual highway |
| Final logits and softmax | Tied embedding transpose, vocabulary softmax, and next-token choice |

The reusable conceptual pieces from the copied source are:

- `src/llm/walkthrough/Walkthrough.ts` for ordered chapters and active-step state
- `src/llm/walkthrough/WalkthroughTools.ts` for timed commentary breaks
- `src/llm/Annotations.ts` for block splitting and index highlights
- `src/llm/components/DataFlow.ts` and `src/llm/Interaction.ts` for cell dependencies
- `src/llm/Commentary.tsx` and `src/llm/Sidebar.tsx` for guided progression
- `src/llm/walkthrough/Walkthrough04_SelfAttention.tsx` for row/column dot-product choreography
- `src/llm/walkthrough/Walkthrough07_Mlp.tsx` for projection, activation, projection, and residual choreography

When moving more of `/qwen` into the WebGL renderer, replace one original matrix block at a time and
keep the `IBlkDeps` dependency graph complete. `drawDependences` and `drawDataFlow` can only trace a
computed cell if every source index mapping is accurate.

## File map

| Path | Responsibility |
|---|---|
| `src/app/qwen/page.tsx` | Next.js `/qwen` route and metadata |
| `src/app/qwen/QwenClient.tsx` | Client-side mount and cleanup |
| `src/app/qwen/layout.tsx` | Route-scoped visualization styles |
| `qwen/math.js` | Matrix and activation primitives |
| `qwen/model.js` | Deterministic mini-model forward pass and recorded intermediates |
| `qwen/tutorial.js` | Sidebar chapters, equations, and visual payloads |
| `qwen/app.js` | Interaction state and DOM/SVG-style renderer |
| `qwen/styles.css` | Layout, arrows, animations, matrices, and color semantics |
| `qwen/index.html` | Dependency-free standalone entry point |
| `llm-viz/` | Original implementation reference only |

## Development

The copied Next.js project keeps its original commands:

```bash
yarn
yarn dev
```

Open `http://localhost:3002/qwen`.

The Qwen visualization also has a dependency-free standalone path for quick iteration:

```bash
python3 -m http.server 3002
```

Open `http://localhost:3002/qwen/`.

Before finishing changes:

```bash
node --check qwen/math.js
node --check qwen/model.js
node --check qwen/tutorial.js
node --check qwen/app.js
node -e "import('./qwen/tutorial.js').then(({QWEN_TUTORIAL}) => console.log(QWEN_TUTORIAL.steps.length))"
```

When Node and installed dependencies satisfy the copied Next.js version, also run the repository's
`yarn typecheck`, `yarn lint`, and `yarn build` commands.
