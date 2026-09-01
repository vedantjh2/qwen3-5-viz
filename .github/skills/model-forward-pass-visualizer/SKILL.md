---
name: model-forward-pass-visualizer
description: Build an interactive, top-down, cell-traceable forward-pass visualization and sidebar tutorial for a neural-network model or architecture. Use when asked to visualize, explain, teach, animate, or reproduce the forward pass of an LLM, transformer, attention model, SSM, Gated DeltaNet, recurrent model, MoE, multimodal model, or another matrix-heavy architecture.
license: MIT
---

# Model Forward-Pass Visualizer

Create a faithful teaching visualization that follows real tensors through a model one operation at
a time. The result should let a student answer:

- What tensor exists at this point?
- Where was it calculated?
- Which learned values are used now?
- How is one selected result cell computed?
- What consumes the result next?
- Which operations are sequential, parallel, recurrent, or repeated?

Use the existing Qwen3.5 visualization in this repository as the implementation reference, but adapt
the mathematics and tutorial structure to the requested architecture rather than renaming Qwen
blocks.

## Read these resources first

Before implementing, read:

- `references/architecture-adapter.md`
- `references/renderer-contract.md`
- `references/quality-checklist.md`
- `examples/qwen3-5.md`
- The target repository's `AGENTS.md`

When working in this repository, also inspect:

- `qwen/math.js`
- `qwen/model.js`
- `qwen/tutorial.js`
- `qwen/app.js`
- `qwen/styles.css`

## Non-negotiable principles

1. **Architecture fidelity over visual convenience.**
   Preserve the real operation graph, ordering, sharing ratios, gates, residual paths, normalization
   placement, state semantics, and tied/untied weights.

2. **Use authoritative sources.**
   Ground the implementation in the released model configuration, official model card, framework
   implementation, and paper or technical report when available. Record URLs and resolved facts in
   `AGENTS.md`.

3. **Miniaturize shapes, not behavior.**
   Use small dimensions that keep cells readable while preserving defining ratios. If a ratio cannot
   be preserved, state the deviation prominently and explain why.

4. **Blue means learned values; green means computed values.**
   Never invert this convention. Learned values include embeddings, projection matrices,
   normalization scales, convolution kernels, and learned decay parameters. Computed values include
   activations, recurrent states, probabilities, logits, and residual streams.

5. **One Continue step, one operation.**
   Do not hide required forward-pass work behind tabs. If several operations can run in parallel,
   announce that fact, then serialize them for teaching.

6. **Animations play once.**
   Animate one aligned multiplication term at a time, in inner-dimension order, then stop on the
   final sum. Replay repeats only the current operation. Never loop indefinitely or automatically
   advance chapters.

7. **Every reused tensor needs provenance.**
   Name the exact earlier tutorial step that produced it. Also state where the current result is
   consumed next.

8. **Explain repeated blocks once.**
   Fully teach one representative repeated block. Numerically execute later identical blocks, but
   collapse their repeated explanation into a fast-forward that emphasizes new parameters, new
   states, and changed activations.

## Required workflow

### 1. Establish the exact target

Resolve:

- Model family and exact checkpoint or release
- Dense versus MoE
- Text-only versus multimodal path
- Prefill versus decode behavior when they differ
- Training-only modules that should not appear during inference
- The repeated architectural unit to visualize

If the request names a family but not a checkpoint, choose a representative released checkpoint and
state the assumption.

### 2. Research the implementation

Use the source hierarchy in `references/architecture-adapter.md`.

Produce an architecture inventory before coding:

- Input and embedding path
- Layer type schedule
- Normalization placement
- Every learned projection
- Every nonlinearity and gate
- Token-mixing operations
- State/cache shapes and update equations
- Residual additions
- FFN or expert routing
- Final normalization and output head
- Weight tying

Do not infer a modern architecture from a similarly named older model.

### 3. Define real and teaching configurations

Add a real-versus-mini fidelity table to `AGENTS.md`.

The teaching configuration should:

- Keep sequence length between 3 and 6 tokens
- Keep hidden width between 6 and 12 when possible
- Preserve head-sharing ratios exactly
- Preserve recurrent state rank and axis meaning
- Preserve convolution kernel length when it is pedagogically useful
- Use a compact FFN width that still shows expansion and contraction
- Keep vocabulary small enough to display the output head

Mark the mini model as a teaching model, not an official checkpoint.

### 4. Implement a deterministic forward pass

Build the mini model in a pure data module modeled after `qwen/model.js`.

Requirements:

- Seed all generated values
- Keep learned tensors separate from computed tensors
- Record every intermediate needed by the tutorial
- Store per-token and per-head recurrent traces
- Preserve tied weights by reusing the same matrix object or values
- Avoid arbitrary post-hoc output overrides

If the user requests an intuitive final token, prefer changing an unused fixed vocabulary row or
another fixed learned value that does not alter prior activations. Calculate that value offline,
hardcode it before the forward pass, and disclose that it is an intentionally designed toy target.

### 5. Build the tutorial sequence

Create steps in actual forward-pass order.

Each step must include:

- Short sidebar title
- Detailed operation title
- Formula
- Real and mini shapes
- Two or more operation-specific explanatory paragraphs
- "Why this operation exists"
- Exact input provenance
- Exact output handoff
- Numbered cell-level execution steps
- Visual payload

For parallel branches, use consecutive steps with an execution-order note such as:

> These projections use the same available input and may execute in parallel. The tutorial shows
> gate first and up second so each matrix multiplication can be inspected independently.

### 6. Handle recurrent and stateful models correctly

Never present recurrent token states as independent tabs.

Show the chain explicitly:

```text
S_-1 --token 0--> S_0 --token 1--> S_1 --token 2--> S_2
```

Token controls may inspect checkpoints, but label them as inspection controls. Explain:

- Which state is shared across time
- Which states are independent across heads, layers, batches, or experts
- Whether optimized prefill uses a chunked but algebraically equivalent algorithm
- How per-token outputs are stacked and concatenated into the next dense tensor

Always add an explicit bridge between a per-token symbol such as `o_t` and the batched tensor such as
`O_core`.

### 7. Trace every matrix result cell

For `C = A x B`, selecting `C[r,c]` must:

1. Highlight row `r` of computed input `A`
2. Highlight column `c` of `B`
3. Animate one pair `A[r,k]`, `B[k,c]` at a time
4. Show each product
5. Apply any explicit scale to each term
6. Show the running sum
7. Stop at the final value

If both operands are computed, both arrows are green. If the right operand is learned, its arrow is
blue.

For non-matmul operations, selecting a result cell must show the exact scalar formula and source
coordinates.

### 8. Build the top-down model view

The main architecture rail should:

- Start with tokens and embeddings
- Follow the forward pass from top to bottom
- Keep the residual stream visually continuous
- Show blue side arrows for learned parameter feeds
- Show green arrows for computed-value flow
- Center the active operation
- Keep repeated but collapsed layers visible
- End with final norm, output head, and next-token distribution

Use query-string deep links such as `?step=layer-3-out-proj`.

### 9. Write durable project documentation

Update `AGENTS.md` with:

- Source URLs
- Exact real configuration
- Teaching configuration and deviations
- Full forward equations
- State/cache semantics
- Color and animation contract
- Tutorial structure
- File map
- Validation commands

Update `README.md` with local and hosted usage.

### 10. Validate before publishing

Follow every item in `references/quality-checklist.md`.

At minimum:

- Recompute every visualized matmul and require zero or negligible error
- Check softmax row sums
- Check unique step IDs
- Check no step contains hidden required operations
- Check architecture ratios
- Check recurrent ordering
- Render representative steps in a real browser
- Verify responsive layout
- Run existing type and syntax checks
- Scan publishable files for secrets

### 11. Publish safely when requested

For GitHub Pages, prefer publishing the dependency-free static visualization directory with official
GitHub Pages actions. Do not require Next.js when the standalone app is sufficient.

Do not add Copilot commit attribution when the user has asked to omit it.

## Deliverables

A completed architecture visualization should contain:

- Deterministic mini-model implementation
- Ordered tutorial data
- Interactive renderer
- Responsive styles
- Architecture and maintenance documentation
- Reusable deep links
- Numerical validation evidence
- Optional GitHub Pages workflow

Stop only after the requested local or hosted experience is working end to end.
