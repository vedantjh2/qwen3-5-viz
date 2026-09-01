# Renderer and Tutorial Contract

The current static implementation is a reusable kit:

| File | Responsibility |
|---|---|
| `qwen/math.js` | Matrix primitives and activations |
| `qwen/model.js` | Deterministic architecture adapter and recorded intermediates |
| `qwen/tutorial.js` | Ordered tutorial steps and visual payloads |
| `qwen/app.js` | Navigation, provenance, rendering, and interaction |
| `qwen/styles.css` | Layout, matrices, arrows, responsive behavior |
| `qwen/index.html` | Dependency-free entry point |

For a new architecture, copy these files into an architecture-named directory or generalize the
existing modules. Do not edit the Qwen equations into misleading generic names.

## Matrix representation

Use:

```js
{
  name: 'attention output weights',
  symbol: 'W_o',
  values: [[...], ...],
  role: 'weight',
  rowLabels: ['h0d0', 'h0d1'],
  colLabels: ['c0', 'c1']
}
```

Supported semantic roles:

- `weight`
- `activation`
- `result`
- `state`
- `gate`
- `probability`
- `mask`

All roles except `weight` are computed and should use the green family. Masks may use neutral gray
while retaining green flow arrows.

## Step representation

Each tutorial step should expose:

```js
{
  id: 'layer-3-out-proj',
  groupId: 'full-attention',
  groupTitle: 'Layer 3: full attention',
  shortTitle: 'Attention O matmul',
  eyebrow: 'Layer 3 / gated attention',
  title: 'Mix gated heads through the output projection',
  architectureTarget: 'layer-3-mixer',
  paragraphs: ['...', '...'],
  formula: 'M = C_gated W_o',
  callouts: [
    ['Mini matmul', '(4 x 8) x (8 x 8)'],
    ['Real weight', '(4096 x 2560)'],
    ['Output', '4 x 8']
  ],
  visual: {
    kind: 'matmul',
    operations: [operation]
  }
}
```

Step IDs must be unique and stable because query-string deep links depend on them.

## Matmul operation

Use:

```js
{
  id: 'layer-3-attention-out',
  title: 'Attention output projection',
  equation: 'M = C_gated x W_o',
  caption: '...',
  left,
  right,
  result,
  defaultCell: [3, 2],
  termScale: 1
}
```

Required behavior:

- Selectable result cells
- Row and column highlighting
- One active inner term
- Product list
- Running sum
- Final value
- One-shot animation
- Explicit scale handling

## Visual kinds

The current renderer includes:

- `tokens`
- `lookup`
- `overview`
- `matmul`
- `transform`
- `elementwise`
- `convolution`
- `delta`
- `delta-repeat`
- `gqa`
- `softmax`
- `prediction`

Add a new visual kind only when an architecture introduces a genuinely different operation. Keep
the data payload serializable and deterministic.

Potential additions:

- `moe-router`
- `expert-dispatch`
- `selective-scan`
- `cross-attention`
- `vision-patches`
- `pooling`

## Provenance

Every computed input should resolve to an earlier producer.

The current renderer fingerprints matrices by shape and values, then searches earlier step outputs.
For sliced, reshaped, repeated, or concatenated tensors, add an explicit architecture-specific note
that names:

- Original step
- Original matrix
- Transformation applied between then and now
- Why the tensor was saved

Examples:

- `z` was projected before the recurrent scan and reused after `O_core` assembly.
- Attention output gates were packed with Q and reused after all context heads.
- FFN `U` was calculated by RMSNorm and reused by both gate and up projections.
- Shared `V0` was projected once and reused by every query head.

## Explanation layout

Render, in order:

1. Execution-order note
2. Base operation narrative
3. "Why this operation exists"
4. Continuity and provenance
5. Forward equation
6. Numbered scalar/cell procedure
7. Shape callouts

The explanation should answer input, purpose, mechanism, output, and next consumer without requiring
the student to remember an unexplained symbol from many steps earlier.

## Repeated blocks

Keep all repeated blocks visible in the architecture rail. Replace duplicate detailed sidebar
chapters with one `repeat` visual showing actual computed matrices between blocks.

Never skip numerical execution merely because tutorial prose is collapsed.

## Accessibility and responsiveness

- Minimum 44px touch targets for primary controls
- Keyboard previous/next and replay controls
- Visible focus rings
- Reduced-motion support
- Horizontal matrix scrolling on narrow screens
- Mobile tutorial drawer
- No information conveyed by color alone
- Text labels for learned and computed arrows
