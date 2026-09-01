# Quality and Publication Checklist

Complete every applicable item before declaring the visualization finished.

## Architecture research

- [ ] Exact checkpoint or release identified
- [ ] Official configuration inspected
- [ ] Exact implementation inspected
- [ ] Layer schedule recorded
- [ ] Dense/MoE distinction confirmed
- [ ] Biases, gates, activations, and norms confirmed
- [ ] Head counts and dimensions confirmed
- [ ] State/cache semantics confirmed
- [ ] Weight tying confirmed
- [ ] Sources recorded in `AGENTS.md`

## Teaching-model fidelity

- [ ] Real-versus-mini table exists
- [ ] Defining head ratios are preserved
- [ ] Residual width is consistent
- [ ] State axes retain their meaning
- [ ] Sequence order is causal
- [ ] Any fidelity deviation is disclosed
- [ ] Toy weights are deterministic
- [ ] Intentionally targeted output is disclosed

## Numerical checks

- [ ] Every matmul has compatible dimensions
- [ ] Every displayed result cell reconstructs from displayed operands
- [ ] Explicit scales are included in reconstruction
- [ ] Element-wise results use matching coordinates
- [ ] Softmax rows sum to one
- [ ] Masked future probabilities are zero
- [ ] Residual additions preserve shape
- [ ] Recurrent states update in token order
- [ ] Tied output weights reuse the embedding matrix

Suggested audit:

```js
for (const operation of allMatmulOperations) {
  const raw = matmul(operation.left.values, operation.right.values);
  const scale = operation.termScale ?? 1;
  assertClose(raw * scale, operation.result.values);
}
```

## Tutorial structure

- [ ] Step IDs are unique
- [ ] Steps follow forward-pass order
- [ ] One required operation per Continue step
- [ ] Parallel branches are announced then serialized
- [ ] Repeated blocks are explained once
- [ ] Stateful token controls are labeled as checkpoints
- [ ] Per-token outputs are explicitly assembled into batched tensors
- [ ] Every reused tensor names its producer
- [ ] Every result names its next consumer
- [ ] Learned parameters are not described as computed activations
- [ ] Computed tensors are not described as pretrained weights

## Cell interaction

- [ ] Result cells are selectable
- [ ] Correct source row is highlighted
- [ ] Correct source column is highlighted
- [ ] Terms animate in inner-dimension order
- [ ] Animation stops after one pass
- [ ] Replay restarts the current calculation
- [ ] Running sum equals final cell
- [ ] Computed/computed matmuls use green arrows for both operands
- [ ] Learned/computed matmuls use a blue weight arrow

## Explanation quality

- [ ] Every step has at least two base paragraphs
- [ ] Every step includes modeling purpose
- [ ] Every step explains relevant shapes
- [ ] Every step states what is and is not mixed
- [ ] Long-range tensor reuse is called out
- [ ] Optimized kernel behavior is distinguished from mathematical semantics
- [ ] Toy behavior is not presented as pretrained capability

## Browser validation

- [ ] Landing step renders
- [ ] Representative matmul renders
- [ ] Recurrent chain renders
- [ ] Recurrent assembly renders
- [ ] Attention sharing renders
- [ ] Mask and softmax render separately
- [ ] Output distribution renders
- [ ] Deep-link query parameters work
- [ ] Desktop screenshot reviewed
- [ ] Mobile screenshot reviewed
- [ ] No `undefined`, `NaN`, or missing labels appear

## Repository checks

- [ ] Existing syntax checks pass
- [ ] Existing type checks pass
- [ ] Authored files pass whitespace checks
- [ ] Publishable tree has no secrets
- [ ] No large accidental model checkpoints are committed
- [ ] Licenses and upstream attribution are preserved
- [ ] Worktree is clean after commit

## GitHub Pages

- [ ] Official GitHub actions only
- [ ] Static visualization directory uploaded directly
- [ ] Pages build type is `workflow`
- [ ] Deployment succeeds
- [ ] Page URL returns HTTP 200
- [ ] JavaScript and CSS assets return HTTP 200
- [ ] Live deep link renders the requested step
- [ ] Commit attribution follows the user's request
