import { formatNumber, transpose } from './math.js';
import {
  MINI_CONFIG,
  REAL_CONFIG,
  makeMatmulOperation,
  matrixData,
} from './model.js';
import { QWEN_TUTORIAL } from './tutorial.js';

let app = null;
const prefersReducedMotion = window.matchMedia(
  '(prefers-reduced-motion: reduce)',
).matches;

const state = {
  stepIndex: 0,
  openGroups: new Set(['orientation']),
  visualTab: 0,
  selectedCell: null,
  activeTerm: 0,
  paused: prefersReducedMotion,
  lookupToken: 0,
  deltaToken: 3,
  deltaHead: 0,
  deltaPhase: 'predict',
  softmaxRow: 3,
  replayKey: 0,
  mobileSidebarOpen: false,
};

let termTimer = null;

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function stepById(stepId) {
  return QWEN_TUTORIAL.steps.find((step) => step.id === stepId);
}

function currentStep() {
  return QWEN_TUTORIAL.steps[state.stepIndex];
}

function roleClass(role) {
  if (role === 'weight') return 'weight-matrix';
  if (role === 'mask') return 'mask-matrix';
  if (role === 'probability') return 'probability-matrix';
  if (role === 'state') return 'state-matrix';
  if (role === 'gate') return 'gate-matrix';
  if (role === 'result') return 'result-matrix';
  return 'computed-matrix';
}

function cellColor(value, role, emphasized) {
  if (!Number.isFinite(value)) {
    return 'oklch(0.91 0.012 260)';
  }
  const strength = Math.min(1, Math.abs(value) / 1.4);
  const hue = role === 'weight' ? 252 : 150;
  const lightness = emphasized ? 0.75 : 0.968 - strength * 0.12;
  const chroma = emphasized ? 0.16 : 0.026 + strength * 0.075;
  return `oklch(${lightness} ${chroma} ${hue})`;
}

function formatCell(value) {
  if (value === Number.NEGATIVE_INFINITY) return '-inf';
  const normalized = Math.abs(value) < 0.005 ? 0 : value;
  if (Math.abs(normalized) >= 10) return normalized.toFixed(1);
  return normalized.toFixed(2);
}

function sameCell(cell, row, col) {
  return cell && cell[0] === row && cell[1] === col;
}

function matrixGrid(data, options = {}) {
  const {
    selectedCell = null,
    activeCells = [],
    highlightRow = null,
    highlightCol = null,
    clickable = false,
    compact = false,
    extraClass = '',
  } = options;
  const activeSet = new Set(activeCells.map(([row, col]) => `${row}:${col}`));
  const rows = data.values.length;
  const cols = data.values[0]?.length ?? 0;

  const columnHeaders = Array.from({ length: cols }, (_, colIndex) => `
    <div class="matrix-axis-label">${escapeHtml(data.colLabels?.[colIndex] ?? colIndex)}</div>
  `).join('');

  const cells = data.values
    .map((row, rowIndex) => {
      const rowCells = row
        .map((value, colIndex) => {
          const selected = sameCell(selectedCell, rowIndex, colIndex);
          const active = activeSet.has(`${rowIndex}:${colIndex}`);
          const vector =
            highlightRow === rowIndex || highlightCol === colIndex;
          const classes = [
            'matrix-cell',
            clickable ? 'clickable-cell' : '',
            selected ? 'selected-cell' : '',
            active ? 'active-term-cell' : '',
            vector ? 'vector-cell' : '',
          ]
            .filter(Boolean)
            .join(' ');
          const attributes = clickable
            ? `data-output-cell data-row="${rowIndex}" data-col="${colIndex}" aria-selected="${selected}"`
            : '';
          const element = clickable ? 'button' : 'div';
          return `
            <${element}
              ${clickable ? 'type="button" role="gridcell"' : ''}
              class="${classes}"
              ${attributes}
              style="background:${cellColor(value, data.role, selected || active)}"
              title="${escapeHtml(`${data.symbol}[${rowIndex},${colIndex}] = ${value}`)}"
            >${escapeHtml(formatCell(value))}</${element}>
          `;
        })
        .join('');
      return `
        <div class="matrix-axis-label matrix-row-label">${escapeHtml(data.rowLabels?.[rowIndex] ?? rowIndex)}</div>
        ${rowCells}
      `;
    })
    .join('');

  return `
    <figure class="matrix-figure ${roleClass(data.role)} ${compact ? 'compact-matrix' : ''} ${extraClass}">
      <figcaption class="matrix-caption">
        <span><strong>${escapeHtml(data.symbol)}</strong>${escapeHtml(data.name)}</span>
        <code>${rows} x ${cols}</code>
      </figcaption>
      <div class="matrix-scroller">
        <div
          class="matrix-grid"
          style="grid-template-columns:minmax(3.8rem,auto) repeat(${cols},var(--matrix-cell))"
          ${clickable ? `role="grid" aria-label="${escapeHtml(`${data.name}; click a green result cell to trace it`)}"` : ''}
        >
          <div class="matrix-axis-label matrix-corner">${escapeHtml(data.symbol)}</div>
          ${columnHeaders}
          ${cells}
        </div>
      </div>
    </figure>
  `;
}

function colorLegend() {
  return `
    <div class="color-legend" aria-label="Visualization color legend">
      <span><i class="blue-swatch"></i>Blue: learned / pretrained weights</span>
      <span><i class="green-swatch"></i>Green: computed values and flow</span>
    </div>
  `;
}

function resetVisualState() {
  const visual = currentStep().visual;
  state.visualTab = 0;
  state.activeTerm = 0;
  state.paused = prefersReducedMotion;
  state.lookupToken = 0;
  state.deltaPhase = visual.phase ?? 'predict';
  state.softmaxRow = visual.defaultRow ?? 3;

  if (visual.kind === 'matmul' || visual.kind === 'transform') {
    state.selectedCell = [...visual.operations[0].defaultCell];
  } else if (
    visual.kind === 'convolution' ||
    visual.kind === 'elementwise'
  ) {
    state.selectedCell = [...visual.defaultCell];
  } else {
    state.selectedCell = null;
  }
}

function setStep(index) {
  const clamped = Math.max(
    0,
    Math.min(QWEN_TUTORIAL.steps.length - 1, index),
  );
  state.stepIndex = clamped;
  const step = currentStep();
  const url = new URL(window.location.href);
  url.searchParams.set('step', step.id);
  window.history.replaceState(null, '', url);
  state.openGroups.add(step.groupId);
  state.mobileSidebarOpen = false;
  resetVisualState();
  renderSidebar();
  renderArchitecture();
  renderOperation();
}

function stepBreakdown(step) {
  const visual = step.visual;
  if (visual.kind === 'matmul') {
    const operation = visual.operations[0];
    const innerLength = operation.left.values[0].length;
    const rightDescription =
      operation.right.role === 'weight'
        ? `the blue learned column from ${operation.right.symbol}`
        : `the green computed column from ${operation.right.symbol}`;
    const items = [
      `Choose one green destination cell in ${operation.result.symbol}; its row and column fix the entire calculation.`,
      `Read that row from ${operation.left.symbol}. It contributes ${innerLength} ordered values.`,
      `Read ${rightDescription}. It contributes the matching ${innerLength} values.`,
      `For k = 0 through ${innerLength - 1}, multiply exactly one aligned pair. The animation highlights only that pair.`,
    ];
    if ((operation.termScale ?? 1) !== 1) {
      items.push(
        `Apply the explicit scale ${formatCell(operation.termScale)} to each product.`,
      );
    }
    items.push(
      `Add the products in order and write the final sum into the selected green ${operation.result.symbol} cell.`,
    );
    if (operation.derived) {
      items.push(
        `Only after the matmul is complete, apply ${operation.derived.formula} element by element.`,
      );
    }
    return items;
  }
  if (visual.kind === 'transform') {
    const operation = visual.operations[0];
    if (operation.input.symbol === 'O_core') {
      return [
        'Read one O_core token row assembled in the immediately preceding step.',
        'Split columns v0/v1 back into value head 0 and columns v2/v3 back into value head 1.',
        'Compute RMSNorm separately inside each two-value head; the heads are not normalized together.',
        'Apply SiLU to the matching z values and multiply them into the normalized head coordinates.',
        'Place the two gated head slices back beside each other to form the O_gated row.',
      ];
    }
    return [
      `Choose one green output coordinate in ${operation.output.symbol}.`,
      `Read the corresponding input value and any full row or blue learned scale required by the formula.`,
      `Evaluate ${operation.formula} without mixing token rows.`,
      `Write the computed value into the same output coordinate, then repeat for the remaining cells.`,
    ];
  }
  if (visual.kind === 'elementwise') {
    return [
      `Choose one green result coordinate [t,c].`,
      `Read the value at [t,c] from ${visual.left.symbol}.`,
      `Read the value at the identical coordinate from ${visual.right.symbol}.`,
      `Apply ${visual.formula} and store the result at [t,c]; no other row or column participates.`,
    ];
  }
  if (visual.kind === 'convolution') {
    return [
      'Choose one token position t and one packed Q/K/V channel c.',
      'Align that channel with its four blue depthwise-kernel taps.',
      'Read only t, t-1, t-2, and t-3; positions before the prompt are zero padding and future positions are forbidden.',
      'Multiply one input/tap pair at a time, add the four products, apply SiLU, and write one green output cell.',
    ];
  }
  if (visual.kind === 'delta') {
    const phaseSteps = {
      chain: [
        'Choose one value head; that head owns one recurrent state matrix.',
        'Start from S_-1, then process t0 to produce S_0 and o_0.',
        'Feed S_0 into t1, then continue through t2 and t3 without resetting state.',
        'Use the time-step controls only to inspect a checkpoint in this chain, not to select an independent calculation.',
      ],
      decay: [
        'Select one token and one value head.',
        'Read the state left by the preceding token.',
        'Multiply every state cell by this token/head retention factor exp(g).',
        'Carry the decayed state into the prediction step.',
      ],
      predict: [
        'Select one output coordinate in the memory vector.',
        'Take the current normalized key row and the matching decayed-state column.',
        'Multiply aligned key/state entries one at a time and add them.',
        'Store the sum as what memory already predicts for that value coordinate.',
      ],
      correct: [
        'Read the current value target and the just-computed memory prediction.',
        'Subtract prediction from target at the same value coordinate.',
        'Multiply that error by beta for this token and value head.',
        'Pass the resulting correction vector to the rank-one write.',
      ],
      write: [
        'Transpose the key from a row into a column.',
        'Choose one state coordinate [key_dimension,value_dimension].',
        'Multiply that key component by the matching correction component.',
        'Write the product into the rank-one update matrix.',
      ],
      update: [
        'Choose one state coordinate.',
        'Read the decayed old-state value at that coordinate.',
        'Read the rank-one write at the same coordinate.',
        'Add them and carry the updated state to both the read step and the next token.',
      ],
      read: [
        'Choose one output value coordinate.',
        'Take the current scaled query row and the matching updated-state column.',
        'Multiply aligned entries one at a time and add them.',
        'Store the sum as this value-head output for the current token.',
      ],
      assemble: [
        'For head 0, stack o_0, o_1, o_2, and o_3 as four rows.',
        'Do the same for head 1 using its separate recurrent state chain.',
        'For each token row, place head 0 values first and head 1 values second.',
        'The concatenated 4 x 4 matrix is O_core, which flows directly into per-head gated RMSNorm.',
      ],
    };
    return phaseSteps[visual.phase];
  }
  if (visual.kind === 'softmax') {
    if (visual.phase === 'mask') {
      return [
        'Choose one query-token row.',
        'Keep scores on and left of the diagonal because they refer to current or earlier tokens.',
        'Replace every score to the right of the diagonal with negative infinity.',
        'Pass the completed masked row to softmax.',
      ];
    }
    return [
      'Choose one causally masked score row.',
      'Subtract its largest finite value for numerical stability.',
      'Exponentiate each finite entry; negative infinity becomes zero.',
      'Add the exponentials and divide each one by the sum so the probability row totals one.',
    ];
  }
  if (visual.kind === 'lookup') {
    return [
      'Read the integer token ID at one sequence position.',
      'Use that ID as a row index into the blue embedding table.',
      'Copy the entire selected blue row.',
      'Place the copied numbers into the green residual-stream row at the same token position.',
    ];
  }
  if (visual.kind === 'tokens') {
    return [
      'Split the prompt into tokenizer pieces.',
      'Map each piece to its vocabulary integer.',
      'Preserve the token order because every later operation is causal.',
      'Pass the integer sequence to embedding lookup.',
    ];
  }
  if (visual.kind === 'gqa') {
    return [
      'Keep query heads distinct.',
      `Assign all ${visual.qHeads} mini query heads to the same key/value group.`,
      'Reuse the shared K and V tensors for each assigned query head.',
      'Compute each head independently in the next score steps.',
    ];
  }
  if (visual.kind === 'delta-repeat') {
    return [
      'Take the fully explained output of Gated DeltaNet layer 0.',
      'Run the same mixer, residual, RMSNorm, SwiGLU, and residual sequence in layer 1 using layer 1 weights and fresh layer 1 state chains.',
      'Feed layer 1 output into layer 2 and repeat with layer 2 weights and fresh layer 2 state chains.',
      'Pass layer 2 output into the detailed full-attention layer.',
    ];
  }
  if (visual.kind === 'prediction') {
    return [
      'Read the final-token vocabulary logit row.',
      'Apply softmax across all vocabulary entries.',
      'Compare the resulting probabilities.',
      'Choose the largest probability for greedy decoding and append that token to the sequence.',
    ];
  }
  return [
    'Identify the current green tensor on the top-down path.',
    'Apply only the operation highlighted in this step.',
    'Keep its result for the next Continue step.',
  ];
}

function matrixFingerprint(matrix) {
  return `${matrix.values.length}x${matrix.values[0]?.length ?? 0}:` +
    matrix.values
      .flat()
      .map((value) =>
        value === Number.NEGATIVE_INFINITY ? '-inf' : value.toFixed(10),
      )
      .join(',');
}

function uniqueMatrices(matrices) {
  const seen = new Set();
  return matrices.filter((matrix) => {
    if (!matrix?.values?.length) return false;
    const key = `${matrix.symbol}:${matrixFingerprint(matrix)}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function producedMatrices(step) {
  const visual = step.visual;
  if (visual.kind === 'matmul') {
    const operation = visual.operations[0];
    return [
      operation.result,
      ...(operation.derived ? [operation.derived.matrix] : []),
    ];
  }
  if (visual.kind === 'transform') {
    return [visual.operations[0].output];
  }
  if (visual.kind === 'elementwise') return [visual.result];
  if (visual.kind === 'convolution') return [visual.output];
  if (visual.kind === 'lookup') return [visual.result];
  if (visual.kind === 'delta-repeat') return visual.stages.slice(1);
  if (visual.kind === 'softmax') {
    return [
      visual.phase === 'mask'
        ? visual.maskedScores
        : visual.probabilities,
    ];
  }
  if (visual.kind === 'prediction') return [visual.probabilities];
  if (visual.kind === 'delta') {
    const fieldByPhase = {
      decay: 'stateDecayed',
      predict: 'memory',
      correct: 'correction',
      write: 'write',
      update: 'stateAfter',
      read: 'read',
    };
    if (visual.phase === 'assemble') return [visual.coreOutput];
    const field = fieldByPhase[visual.phase];
    return field ? visual.traces.map((trace) => trace[field]) : [];
  }
  return [];
}

function consumedMatrices(step) {
  const visual = step.visual;
  if (visual.kind === 'matmul') {
    const operation = visual.operations[0];
    return [operation.left, operation.right];
  }
  if (visual.kind === 'transform') {
    const operation = visual.operations[0];
    return [operation.input, ...(operation.auxiliaries ?? [])];
  }
  if (visual.kind === 'elementwise') return [visual.left, visual.right];
  if (visual.kind === 'convolution') return [visual.input, visual.kernels];
  if (visual.kind === 'lookup') return [visual.embeddingTable];
  if (visual.kind === 'delta-repeat') return [visual.stages[0]];
  if (visual.kind === 'gqa') return [...visual.q, ...visual.k, ...visual.v];
  if (visual.kind === 'softmax') {
    return [
      visual.phase === 'mask' ? visual.scores : visual.maskedScores,
    ];
  }
  if (visual.kind === 'prediction') return [visual.logits];
  if (visual.kind === 'delta') {
    const fieldsByPhase = {
      decay: ['stateBefore'],
      predict: ['k', 'stateDecayed'],
      correct: ['v', 'memory'],
      write: ['k', 'correction'],
      update: ['stateDecayed', 'write'],
      read: ['q', 'stateAfter'],
      assemble: ['headOutputs'],
    };
    const fields = fieldsByPhase[visual.phase] ?? [];
    if (fields.includes('headOutputs')) return visual.headOutputs;
    return visual.traces.flatMap((trace) =>
      fields.map((field) => trace[field]),
    );
  }
  return [];
}

function findProducer(matrix, stepIndex) {
  const fingerprint = matrixFingerprint(matrix);
  for (let index = stepIndex - 1; index >= 0; index -= 1) {
    const match = producedMatrices(QWEN_TUTORIAL.steps[index]).find(
      (candidate) => matrixFingerprint(candidate) === fingerprint,
    );
    if (match) {
      return {
        index,
        step: QWEN_TUTORIAL.steps[index],
        matrix: match,
      };
    }
  }
  return null;
}

function findConsumer(matrix, stepIndex) {
  const fingerprint = matrixFingerprint(matrix);
  for (
    let index = stepIndex + 1;
    index < QWEN_TUTORIAL.steps.length;
    index += 1
  ) {
    const match = consumedMatrices(QWEN_TUTORIAL.steps[index]).find(
      (candidate) => matrixFingerprint(candidate) === fingerprint,
    );
    if (match) {
      return {
        index,
        step: QWEN_TUTORIAL.steps[index],
        matrix: match,
      };
    }
  }
  return null;
}

function specialOriginNotes(step) {
  const visual = step.visual;
  if (visual.kind === 'delta') {
    const notes = {
      chain:
        'The Q, K, and V channels shown in this recurrence were created by the packed QKV projection and then modified by the causal depthwise convolution.',
      decay:
        'The retention factor exp(g_t) comes from the earlier Decay g projection; S_(t-1) is zero only at t0 and otherwise is the updated state from the preceding token.',
      predict:
        'k_t is the key slice of the convolved QKV tensor, after L2 normalization. S_tilde was produced in the immediately preceding decay step.',
      correct:
        'v_t is the value slice of the convolved QKV tensor. beta_t was computed earlier by the Write rate beta projection, while memory came from the preceding k_t S_tilde matmul.',
      write:
        'The same normalized k_t used for prediction is reused here. delta_t is the beta-scaled error calculated in the preceding correction step.',
      update:
        'Both inputs were just computed for this checkpoint: S_tilde in the decay step and the rank-one write in the preceding outer-product step.',
      read:
        'q_t is the query slice of the convolved QKV tensor after L2 normalization and 1/sqrt(d_k) scaling. S_t was completed in the preceding update step.',
      assemble:
        'Every row in O_head0 and O_head1 was produced by the q_t S_t read step, repeated along the single recurrent chain for t0 through t3.',
    };
    return [notes[visual.phase]];
  }
  if (visual.kind === 'gqa') {
    return [
      'Q0-Q3 come from the four earlier Q-head RMSNorm/RoPE steps. K0 comes from the Shared K RoPE step, and V0 comes from the earlier V projection.',
    ];
  }
  if (visual.kind === 'delta-repeat') {
    return [
      "X_after_L0 is the second residual result at the end of the fully explained layer 0. X_after_L1 and X_after_L2 are genuine outputs from running the same graph with the next layers' parameters.",
    ];
  }
  if (
    visual.kind === 'transform' &&
    visual.operations[0].input.symbol === 'O_core'
  ) {
    return [
      'O_core was assembled in the immediately preceding step by stacking every o_t read for each value head and concatenating the two head matrices.',
      'The z matrix was calculated much earlier in the Output gate z projection, before the recurrent scan began; it is intentionally reused here.',
    ];
  }
  if (step.id.includes('-q-rope-')) {
    return [
      'This Q-head slice comes from the packed Q + output-gate projection. The packed result was split into four two-dimensional query heads before this normalization and rotation.',
    ];
  }
  if (step.id.includes('-scores-head')) {
    return [
      'The selected Q head was normalized and rotated in its earlier Q-head RoPE step. The shared K0 matrix was normalized and rotated once in the Shared K step.',
    ];
  }
  if (step.id.includes('-context-head')) {
    return [
      'P_h is the probability matrix produced by this heads causal-mask and row-softmax steps. V0 was computed earlier by the shared V projection and is reused by every query head.',
    ];
  }
  if (step.id.endsWith('-attn-gate')) {
    return [
      'The four context matrices were produced by the four preceding P_h x V0 matmuls and concatenated here. The gate logits were saved from the much earlier packed Q + output-gate projection.',
    ];
  }
  return [];
}

function stepContext(stepIndex) {
  const step = QWEN_TUTORIAL.steps[stepIndex];
  const previous = QWEN_TUTORIAL.steps[stepIndex - 1];
  const next = QWEN_TUTORIAL.steps[stepIndex + 1];
  const inputs = uniqueMatrices(consumedMatrices(step));
  const origins = [];

  for (const matrix of inputs) {
    if (matrix.role === 'weight') {
      origins.push(
        `${matrix.symbol} is a blue learned-parameter slot, not a forward-pass result. A real checkpoint loads it from model weights; this teaching model uses a deterministic stand-in with the same shape and role.`,
      );
      continue;
    }
    const producer = findProducer(matrix, stepIndex);
    if (producer) {
      origins.push(
        `${matrix.symbol} was computed in step ${producer.index + 1}, "${producer.step.title}", where it appeared as ${producer.matrix.symbol}.`,
      );
    }
  }

  origins.push(...specialOriginNotes(step).filter(Boolean));
  const dedupedOrigins = [...new Set(origins)];
  if (dedupedOrigins.length === 0) {
    dedupedOrigins.push(
      previous
        ? `This step continues directly from "${previous.title}".`
        : 'This is the start of the forward pass, so there is no earlier computed matrix.',
    );
  }

  const destinations = [];
  for (const matrix of uniqueMatrices(producedMatrices(step))) {
    const consumer = findConsumer(matrix, stepIndex);
    if (consumer) {
      destinations.push(
        `${matrix.symbol} is next reused in step ${consumer.index + 1}, "${consumer.step.title}".`,
      );
    }
  }

  return {
    transition: previous
      ? `You arrived here from step ${stepIndex}, "${previous.title}".`
      : 'This step begins the forward pass from tokenizer output.',
    origins: dedupedOrigins,
    handoff:
      [...new Set(destinations)][0] ??
      (next
        ? `After this operation, Continue moves to "${next.title}".`
        : 'This is the final tutorial output.'),
  };
}

function additionalContextParagraphs(step) {
  const id = step.id;

  if (id === 'tokens') {
    return [
      'A token ID is only an address into learned tables; its numeric magnitude has no meaning. ID 4 is not “more” than ID 2, and arithmetic is never performed directly on these integers.',
      'Keeping the four positions ordered matters because both DeltaNet and full attention are causal. Every later token may use earlier positions, but no operation may let an earlier token read a later one.',
    ];
  }
  if (id === 'embedding') {
    return [
      'The lookup copies a complete hidden-width row without averaging it with any other vocabulary row. Repeated occurrences of the same token start from the same embedding and become context-dependent only after decoder layers process them.',
      'This blue table is tied to the final language-model head, so the same learned vocabulary geometry appears once when tokens enter and again, transposed, when logits are produced.',
    ];
  }
  if (id === 'quartet-overview') {
    return [
      'The three recurrent mixers provide fixed-size causal memory at every position, while the fourth layer periodically performs explicit all-pairs attention. The combination gives the model both efficient long-sequence processing and a direct token-to-token comparison mechanism.',
      'Only the first Gated DeltaNet layer is expanded cell by cell in this tutorial. Layers 1 and 2 still run numerically, then the full-attention layer is expanded because it introduces a genuinely different computation graph.',
    ];
  }
  if (id === 'layer-0-mixer-norm') {
    return [
      'Pre-normalization keeps the scale entering the token mixer predictable while leaving an untouched residual copy beside it. If the mixer produces a poor or small update, the original representation still has a direct path forward.',
      'Because RMSNorm works within one token row, this step cannot move information between “A,” “tiny,” “robot,” and “learns.” Sequence mixing begins only after Q/K/V projection and the causal convolution.',
    ];
  }
  if (id === 'layer-0-qkv') {
    return [
      'Packing Q, K, and V into one matrix multiplication is an implementation efficiency: the mathematical result is the same as three projections whose blue weight matrices were concatenated side by side.',
      'The split is asymmetric. The mini model creates one two-dimensional Q/K head but two two-dimensional value heads, matching Qwen3.5’s choice to provide more value-state capacity than key-address capacity.',
    ];
  }
  if (id === 'layer-0-z-proj') {
    return [
      'z is calculated before the recurrent scan but deliberately saved until after all q_t S_t reads have been assembled into O_core. It therefore gates what leaves DeltaNet rather than changing what is stored in the recurrent state.',
      'Its four columns line up exactly with flattened value channels v0-v3. Later, SiLU(z) multiplies those matching channels after each value head has been RMS-normalized.',
    ];
  }
  if (id === 'layer-0-beta-proj') {
    return [
      'beta acts like a per-token write strength for each value head. It is not the optimizer learning rate used during model training; it is an activation computed during this forward pass.',
      'A beta near zero leaves the decayed state mostly unchanged, while a beta near one writes most of the difference between the current value target and what memory already predicted.',
    ];
  }
  if (id === 'layer-0-decay-proj') {
    return [
      'The transformation guarantees g is non-positive, making exp(g) a retention factor between zero and one. This prevents the recurrence from amplifying the old state merely through the decay path.',
      'Each token and value head receives its own retention amount, so one head can preserve older information while another forgets more aggressively at the same sequence position.',
    ];
  }
  if (id === 'layer-0-conv') {
    return [
      'This short convolution gives Q, K, and V an immediate four-token local context before they interact with recurrent memory. It is useful for patterns where nearby order matters strongly, such as phrase boundaries or local syntax.',
      'Because it is depthwise, a q0 channel is filtered only with the q0 kernel; it is not mixed with q1, k, or v channels here. Cross-channel mixing already happened in the projection and happens again in later output projections.',
    ];
  }
  if (id.endsWith('-scan-chain')) {
    return [
      'The state matrix is best viewed as a small associative memory owned by one value head. Keys determine where corrections are written, and queries determine how that accumulated memory is read.',
      'Optimized prefill kernels may reorganize this recurrence into chunks of matrix operations, but the causal dependency remains identical: the effective state used at token t contains updates from tokens 0 through t-1.',
    ];
  }
  if (id.endsWith('-scan-decay')) {
    return [
      'Decay is applied before the current token reads or writes memory. That ordering means the model first decides how much old information survives, then compares the surviving memory with the new value target.',
      'Within the selected value head, the same scalar retention factor multiplies all four cells of its 2 x 2 mini state. A different token or head can use a different factor.',
    ];
  }
  if (id.endsWith('-scan-predict')) {
    return [
      'k_t S_tilde is a content-addressed lookup: the key components weight the rows of the memory state, producing the value vector that memory associates with this key.',
      'This prediction is not yet the head output. It exists so the delta rule can measure what memory got wrong and write only that error instead of repeatedly storing the full value.',
    ];
  }
  if (id.endsWith('-scan-correct')) {
    return [
      'Subtracting memory from v_t implements the defining delta-rule idea: preserve what is already predicted correctly and focus the update on the residual error.',
      'beta then controls how much of that error is trusted at this token. The resulting delta vector supplies the value-axis side of the next rank-one outer product.',
    ];
  }
  if (id.endsWith('-scan-write')) {
    return [
      'The outer product expands one key column and one correction row into a full state-shaped update. It is rank one, so the write is structured rather than an arbitrary replacement of every state cell.',
      'Key components determine which memory rows receive the correction, while delta components determine what is written along the value columns.',
    ];
  }
  if (id.endsWith('-scan-update')) {
    return [
      'This addition commits the current token’s correction. The result is S_t, which serves two roles: it is read immediately by q_t and then carried forward as the old state for token t+1.',
      'No state is shared between decoder layers. When the next Gated DeltaNet layer begins, it owns a separate set of value-head memories even though it follows the same update rule.',
    ];
  }
  if (id.endsWith('-scan-read')) {
    return [
      'The query can differ from the key: k_t described where the current value should be corrected, while q_t describes what information this token wants to retrieve for its output.',
      'The read happens after the current write, so o_t can include information contributed by the current token as well as all causally retained earlier tokens.',
    ];
  }
  if (id.endsWith('-scan-assemble')) {
    return [
      'Stacking along the row axis restores the ordinary batch-by-time tensor layout expected by the rest of the decoder. Concatenating along columns preserves which value head produced each channel.',
      'No learned weights or sums are involved in this assembly. It is a reshape-and-concatenate boundary between the recurrent head-local calculation and the following dense output path.',
    ];
  }
  if (id === 'layer-0-gated-norm') {
    return [
      'Head-local RMSNorm prevents a high-magnitude recurrent read from dominating simply because its state became numerically larger. Each two-channel head is normalized independently before the heads are rejoined.',
      'Multiplying by SiLU(z) restores learned, token-specific amplitude control. Normalization determines a stable direction; z determines how strongly each normalized channel is allowed to leave the mixer.',
    ];
  }
  if (id === 'layer-0-out-proj') {
    return [
      'O_gated is four channels wide in the mini model because it contains two two-dimensional value heads. W_out maps those head channels into all eight residual-stream channels for each token independently.',
      'This projection mixes features across value heads but does not mix token positions; the recurrent scan already performed causal sequence mixing. Its output M is an update that joins the untouched residual branch next.',
    ];
  }
  if (id === 'layer-0-mixer-residual') {
    return [
      'The residual addition combines two paths with different roles: X_in preserves everything available before DeltaNet, while M contributes the newly retrieved and transformed recurrent-memory information.',
      'Keeping the width unchanged at eight allows the result to feed the FFN and every later decoder layer without another structural conversion.',
    ];
  }
  if (id.endsWith('-ffn-norm')) {
    return [
      'This second pre-norm separates the statistics seen by the FFN from those seen by the token mixer. The residual value itself remains untouched on the bypass path until the FFN update is ready.',
      'The FFN is position-wise: all four token rows share the same blue matrices, but each row is transformed independently with no cross-token reads.',
    ];
  }
  if (id.endsWith('-gate-proj') && id !== 'layer-3-q-gate') {
    return [
      'This branch creates the multiplicative control signal for SwiGLU. Its output is not useful alone; it becomes a smooth gate only after SiLU and then modulates the separately computed up branch.',
      'The gate and up projections read the same normalized U matrix and can be evaluated in parallel, but the tutorial serializes them so every dot product remains visible.',
    ];
  }
  if (id.endsWith('-up-proj')) {
    return [
      'The up branch creates the candidate features that carry content through the FFN. It reuses U from the FFN RMSNorm rather than consuming the gate branch output.',
      'Its wider twelve-channel representation gives the model room to form combinations that do not fit directly in the eight-channel residual stream.',
    ];
  }
  if (id.endsWith('-ffn-swiglu')) {
    return [
      'SwiGLU is multiplicative: each activated gate controls the matching candidate feature. This lets the network conditionally pass or suppress features rather than applying the same nonlinearity to a single projection.',
      'The operation remains token-local and channel-local at this point; the following down projection is what mixes the twelve gated features into new residual channels.',
    ];
  }
  if (id.endsWith('-ffn-down')) {
    return [
      'W_down is the learned compression back to residual width. Each output channel can combine every gated intermediate feature, so this is where the FFN’s expanded representation is synthesized into a usable update.',
      'Like the other FFN operations, it does not mix token positions. The resulting D matrix has exactly the same shape as the residual path so they can be added directly.',
    ];
  }
  if (id.endsWith('-ffn-residual')) {
    return [
      'This second residual addition completes the decoder layer. The token-mixer result is preserved on the left path, while D contributes the position-wise nonlinear feature transformation.',
      'The completed matrix becomes the next decoder layer’s input with no change in sequence length or hidden width.',
    ];
  }
  if (id === 'repeat-delta-layers') {
    return [
      'Although their diagrams are collapsed, layers 1 and 2 still receive different inputs and therefore produce different Q/K/V activations, state trajectories, and residual updates.',
      'The state resets are per layer, not per token: each repeated layer begins its own head memories at the sequence boundary and evolves them across the same four token positions.',
    ];
  }
  if (id === 'layer-3-mixer-norm') {
    return [
      'The input here is the actual layer-2 residual output from the fast-forward step. RMSNorm prepares that matrix for attention while the unnormalized copy waits on the residual branch.',
      'Unlike DeltaNet, this mixer will explicitly compare token positions through T x T score matrices, but the normalization itself is still row-local.',
    ];
  }
  if (id === 'layer-3-q-gate') {
    return [
      'The packed result contains eight query channels arranged as four two-dimensional heads plus eight matching gate channels. Packing them together saves a separate large projection without changing the mathematics.',
      'The query slices are used soon in QK^T scores. The gate slices are held much longer and reused only after all four attention contexts have been concatenated.',
    ];
  }
  if (id === 'layer-3-k') {
    return [
      'Only one two-dimensional K head is produced in the mini model. All four Q heads compare against this same K matrix, which is the defining 4:1 grouped-query sharing pattern.',
      'Reducing K heads reduces the amount of key data cached during autoregressive decoding; distinct query projections still let the four Q heads ask different questions.',
    ];
  }
  if (id === 'layer-3-v') {
    return [
      'V contains the payload that attention will blend after probabilities are known. It is not used to decide relevance; Q and K perform that scoring job.',
      'The same V0 matrix is reused by all four query heads, but each head will combine it with a different probability matrix and therefore produce a different context.',
    ];
  }
  if (/layer-3-q-rope-\d/.test(id)) {
    const head = id.at(-1);
    return [
      `RMSNorm makes query head ${head}'s scale comparable across tokens before position is injected. RoPE then rotates its two visible dimensions by a token-dependent angle without changing vector length.`,
      `This head remains distinct because it came from its own Q projection slice. Its rotated rows will form the left operand of the later Scores_${head} = Q_${head} K_0^T matmul.`,
    ];
  }
  if (id === 'layer-3-k-rope') {
    return [
      'The shared key head is normalized and position-rotated once, then reused unchanged by all four query heads. This preserves shared cache storage while still giving every key row positional phase.',
      'RoPE changes Q/K dot products according to relative position; it does not touch V, which remains the content payload mixed later.',
    ];
  }
  if (id === 'layer-3-gqa') {
    return [
      'Grouped-query attention shares only K and V. Q0-Q3 remain separate and will generate four different score and probability matrices against the same cached keys.',
      'The 4:1 teaching ratio exactly matches the real model’s 16:4 grouping, so each real K/V head likewise serves four query heads.',
    ];
  }
  if (/layer-3-scores-head\d/.test(id)) {
    const head = id.at(-1);
    return [
      `Scores_${head}[t,j] measures how strongly query position t in head ${head} matches key position j before causality and normalization are enforced. It is a similarity-like logit, not yet a probability.`,
      'This is where full attention mixes token axes conceptually: every query row is compared with every key row. The next step removes future columns before softmax can use them.',
    ];
  }
  if (/layer-3-softmax-head\d-mask/.test(id)) {
    return [
      'Using negative infinity instead of zero is important because softmax exponentiates its inputs: exp(-infinity) becomes exactly zero, while exp(0) would still receive probability mass.',
      'The mask changes which positions are legal but does not renormalize the surviving scores. That conversion into a distribution is deliberately separated into the next step.',
    ];
  }
  if (/layer-3-softmax-head\d-normalize/.test(id)) {
    return [
      'Softmax preserves score ordering while converting arbitrary logits into positive weights. Subtracting the row maximum changes none of the final ratios but avoids unnecessarily large exponentials.',
      'Each query row is normalized independently, so every token gets its own distribution over legal key positions for this head.',
    ];
  }
  if (/layer-3-context-head\d/.test(id)) {
    const head = id.at(-1);
    return [
      `This is the actual token-mixing output for attention head ${head}. Each context row is a weighted sum of shared V0 rows, using that query token’s P${head} probabilities as coefficients.`,
      'Although all heads read the same V0 payload, their independently projected queries produced different weights. Concatenating the four resulting contexts therefore preserves four different views of the sequence.',
    ];
  }
  if (id === 'layer-3-attn-gate') {
    return [
      'The four two-dimensional context matrices are first placed side by side, restoring an eight-channel per-token representation. The saved gate logits from the packed Q projection align one-to-one with these channels.',
      'Sigmoid bounds each gate between zero and one, letting the layer suppress particular head channels before they are mixed by W_o.',
    ];
  }
  if (id === 'layer-3-out-proj') {
    return [
      'C_gated already contains the attention result: four two-dimensional head contexts were concatenated after P_h x V0 token mixing and then modulated by their saved sigmoid gates. W_o now lets every residual output channel combine information from every head channel.',
      'This projection is applied independently to each token row, so it does not perform another round of token-to-token attention; that happened in the score, softmax, and P_h x V0 steps. The mini shape is 8 x 8, while Qwen3.5-4B maps 4,096 concatenated head channels to the 2,560-wide residual stream.',
      'The resulting M matrix is still only an attention update. The next step adds it to the untouched layer input carried around the entire attention mixer.',
    ];
  }
  if (id === 'layer-3-mixer-residual') {
    return [
      'The left operand preserves the layer-2 representation that entered attention, while M contributes the newly gathered token-to-token context. Their matching shape makes the merge a direct coordinate-wise addition.',
      'After this point the full-attention mixer is complete. The decoder layer still has its dense SwiGLU FFN half before producing its final output.',
    ];
  }
  if (id === 'final-norm') {
    return [
      'In the real 32-layer model this normalization occurs only after all eight quartets have finished. The teaching model reaches it after one quartet so the same output-head mechanics can be inspected at readable scale.',
      'Final RMSNorm does not choose a token or vocabulary item; it standardizes each hidden row so the following tied-embedding dot products have controlled magnitude.',
    ];
  }
  if (id === 'lm-head') {
    return [
      'Multiplying by E^T compares each hidden row with every vocabulary embedding direction. The outputs are logits: larger values indicate stronger compatibility but are neither bounded nor normalized.',
      'All token rows can produce training logits, but autoregressive generation normally needs only the last prompt row to choose the next token.',
    ];
  }
  if (id === 'next-token') {
    return [
      'Greedy decoding selects the largest probability deterministically. Real generation may instead sample, apply temperature, restrict to top-k/top-p candidates, or enforce structured decoding constraints.',
      'After a token is chosen, it is appended to the sequence and the model runs again using cached recurrent and attention state. Here the fixed toy "fast" embedding is deliberately aligned with the final hidden row so the example is intuitive, while the decoding mechanics remain unchanged.',
    ];
  }
  return [
    'This operation preserves the model’s causal ordering and hidden-width contract while preparing a specifically shaped result for the following step.',
  ];
}

function renderShell() {
  app.innerHTML = `
    <div class="page-shell">
      <header class="top-header">
        <div class="brand-mark" aria-hidden="true">
          <span>Q</span><b>3.5</b>
        </div>
        <div class="header-title">
          <p>Forward-pass microscope</p>
          <h1>Qwen3.5 dense quartet</h1>
        </div>
        <div class="header-meta">
          <span>3 DeltaNet</span>
          <i></i>
          <span>1 full attention</span>
          <button type="button" id="mobile-sidebar-toggle">Tutorial</button>
        </div>
      </header>
      <div class="app-shell">
        <aside class="tutorial-sidebar" id="tutorial-sidebar"></aside>
        <main class="main-view">
          <div class="main-toolbar">
            <div>
              <span class="live-dot"></span>
              <strong>Qwen3.5 Mini</strong>
              <span>deterministic teaching weights, not a checkpoint</span>
            </div>
            ${colorLegend()}
          </div>
          <div class="workspace">
            <section class="architecture-pane">
              <div class="pane-heading">
                <p>Top-down data path</p>
                <span>Arrows follow the forward pass</span>
              </div>
              <div class="architecture-scroll" id="architecture-scroll"></div>
            </section>
            <section class="operation-pane" id="operation-pane"></section>
          </div>
        </main>
      </div>
    </div>
  `;

  document
    .querySelector('#mobile-sidebar-toggle')
    .addEventListener('click', () => {
      state.mobileSidebarOpen = !state.mobileSidebarOpen;
      document
        .querySelector('#tutorial-sidebar')
        .classList.toggle('mobile-open', state.mobileSidebarOpen);
    });
}

function renderSidebar() {
  const sidebar = document.querySelector('#tutorial-sidebar');
  const step = currentStep();
  const context = stepContext(state.stepIndex);
  const moreContext = additionalContextParagraphs(step);
  const progress =
    ((state.stepIndex + 1) / QWEN_TUTORIAL.steps.length) * 100;

  const groupsHtml = QWEN_TUTORIAL.groups
    .map((group) => {
      const open = state.openGroups.has(group.id);
      const groupSteps = group.stepIds.map(stepById);
      const activeInGroup = groupSteps.some(
        (item) => item.id === step.id,
      );
      const finished = groupSteps.every(
        (item) =>
          QWEN_TUTORIAL.steps.findIndex((candidate) => candidate.id === item.id) <
          state.stepIndex,
      );
      const stepButtons = groupSteps
        .map((item) => {
          const itemIndex = QWEN_TUTORIAL.steps.findIndex(
            (candidate) => candidate.id === item.id,
          );
          const active = itemIndex === state.stepIndex;
          const complete = itemIndex < state.stepIndex;
          return `
            <button
              type="button"
              class="toc-step ${active ? 'active' : ''} ${complete ? 'complete' : ''}"
              data-step-index="${itemIndex}"
              id="${active ? 'active-toc-step' : ''}"
            >
              <i>${active ? '>' : complete ? '✓' : itemIndex + 1}</i>
              <span>${escapeHtml(item.shortTitle)}</span>
            </button>
          `;
        })
        .join('');
      return `
        <section class="toc-group ${activeInGroup ? 'active-group' : ''}">
          <button type="button" class="toc-group-header" data-group-id="${group.id}">
            <span>
              <strong>${escapeHtml(group.title)}</strong>
              <small>${escapeHtml(group.subtitle)}</small>
            </span>
            <b>${finished ? '✓' : open ? '−' : '+'}</b>
          </button>
          <div class="toc-group-steps ${open ? 'open' : ''}">
            ${stepButtons}
          </div>
        </section>
      `;
    })
    .join('');

  sidebar.classList.toggle('mobile-open', state.mobileSidebarOpen);
  sidebar.innerHTML = `
    <div class="sidebar-top">
      <div class="model-label">
        <span>TEACHING MODEL</span>
        <strong>Qwen3.5 Mini</strong>
        <small>Same operation graph, inspectable dimensions</small>
      </div>
      <div class="progress-meta">
        <span>Step ${state.stepIndex + 1} of ${QWEN_TUTORIAL.steps.length}</span>
        <span>${Math.round(progress)}%</span>
      </div>
      <div class="progress-track"><i style="width:${progress}%"></i></div>
    </div>
    <article class="commentary-card">
      <p class="commentary-eyebrow">${escapeHtml(step.eyebrow)}</p>
      <h2>${escapeHtml(step.title)}</h2>
      ${
        step.sequenceNote
          ? `<div class="sequence-note"><span>Execution order</span><p>${escapeHtml(step.sequenceNote)}</p></div>`
          : ''
      }
      ${step.paragraphs.map((paragraph) => `<p>${escapeHtml(paragraph)}</p>`).join('')}
      <div class="more-context">
        <span>Why this operation exists</span>
        ${moreContext
          .map((paragraph) => `<p>${escapeHtml(paragraph)}</p>`)
          .join('')}
      </div>
      <div class="provenance-card">
        <span>Continuity and provenance</span>
        <p>${escapeHtml(context.transition)}</p>
        <strong>Where the inputs came from</strong>
        <ul>
          ${context.origins
            .map((origin) => `<li>${escapeHtml(origin)}</li>`)
            .join('')}
        </ul>
        <div>
          <b>What happens to this result next</b>
          <p>${escapeHtml(context.handoff)}</p>
        </div>
      </div>
      ${
        step.formula
          ? `<div class="sidebar-formula"><span>Forward equation</span><code>${escapeHtml(step.formula)}</code></div>`
          : ''
      }
      <div class="step-breakdown">
        <span>What happens in this step</span>
        <ol>
          ${stepBreakdown(step)
            .map((item) => `<li>${escapeHtml(item)}</li>`)
            .join('')}
        </ol>
      </div>
      <div class="sidebar-callouts">
        ${step.callouts
          .map(
            ([label, value]) => `
              <div><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>
            `,
          )
          .join('')}
      </div>
    </article>
    <div class="sidebar-controls">
      <button type="button" data-nav="-1" ${state.stepIndex === 0 ? 'disabled' : ''}>Previous</button>
      <button type="button" class="replay-button" id="replay-button">Replay</button>
      <button
        type="button"
        class="next-button"
        data-nav="1"
        ${state.stepIndex === QWEN_TUTORIAL.steps.length - 1 ? 'disabled' : ''}
      >Continue</button>
    </div>
    <div class="keyboard-hint">
      <span><kbd>←</kbd><kbd>→</kbd> steps</span>
      <span><kbd>R</kbd> replay calculation</span>
    </div>
    <nav class="toc" aria-label="Forward-pass tutorial chapters">
      ${groupsHtml}
    </nav>
  `;

  sidebar.querySelectorAll('[data-nav]').forEach((button) => {
    button.addEventListener('click', () => {
      setStep(state.stepIndex + Number(button.dataset.nav));
    });
  });
  sidebar.querySelector('#replay-button').addEventListener('click', () => {
    state.replayKey += 1;
    resetVisualState();
    renderVisualOnly();
  });
  sidebar.querySelectorAll('[data-group-id]').forEach((button) => {
    button.addEventListener('click', () => {
      const groupId = button.dataset.groupId;
      if (state.openGroups.has(groupId)) {
        state.openGroups.delete(groupId);
      } else {
        state.openGroups.add(groupId);
      }
      renderSidebar();
    });
  });
  sidebar.querySelectorAll('[data-step-index]').forEach((button) => {
    button.addEventListener('click', () => {
      setStep(Number(button.dataset.stepIndex));
    });
  });

  window.requestAnimationFrame(() => {
    sidebar
      .querySelector('#active-toc-step')
      ?.scrollIntoView({ block: 'nearest' });
  });
}

function flowArrow(label = '') {
  return `
    <div class="vertical-flow computed-flow" aria-hidden="true">
      <i></i>
      ${label ? `<span>${escapeHtml(label)}</span>` : ''}
    </div>
  `;
}

function architectureNode(target, title, subtitle, kind = 'computed', sideWeight = '') {
  const active = currentStep().architectureTarget === target;
  return `
    <div class="architecture-node ${kind} ${active ? 'active' : ''}" data-target="${target}">
      ${sideWeight ? `<div class="weight-feed"><span>${escapeHtml(sideWeight)}</span><i></i></div>` : ''}
      <strong>${escapeHtml(title)}</strong>
      <span>${escapeHtml(subtitle)}</span>
    </div>
  `;
}

function layerCard(layerIndex, kind) {
  const prefix = `layer-${layerIndex}`;
  const activeTarget = currentStep().architectureTarget;
  const active = activeTarget.startsWith(prefix);
  const mixerTitle =
    kind === 'delta'
      ? 'Gated DeltaNet'
      : 'Gated grouped-query attention';
  const mixerSubtitle =
    kind === 'delta'
      ? 'causal conv + fixed recurrent state'
      : 'partial RoPE + QK^T + softmax + PV';

  return `
    <article class="layer-card ${kind} ${active ? 'active-layer' : ''}">
      <header>
        <span>DECODER LAYER ${layerIndex}</span>
        <strong>${kind === 'delta' ? 'SSM-like recurrent mixer' : 'Full attention mixer'}</strong>
      </header>
      <div class="layer-path">
        ${architectureNode(`${prefix}-norm1`, 'RMSNorm', 'pre-mixer normalization')}
        ${flowArrow()}
        ${architectureNode(`${prefix}-mixer`, mixerTitle, mixerSubtitle, 'computed', kind === 'delta' ? 'Wqkv, Wz, Wb, Wa, Wout' : 'Wqg, Wk, Wv, Wo')}
        ${flowArrow()}
        ${architectureNode(`${prefix}-resid1`, 'Residual add', 'X + token-mixer update')}
        ${flowArrow()}
        ${architectureNode(`${prefix}-norm2`, 'RMSNorm', 'pre-FFN normalization')}
        ${flowArrow()}
        ${architectureNode(`${prefix}-ffn`, 'Dense SwiGLU FFN', 'gate/up -> SiLU x -> down', 'computed', 'Wgate, Wup, Wdown')}
        ${flowArrow()}
        ${architectureNode(`${prefix}-resid2`, 'Residual add', 'layer output')}
      </div>
    </article>
  `;
}

function renderArchitecture() {
  const container = document.querySelector('#architecture-scroll');
  const cycleActive = currentStep().architectureTarget === 'cycle';
  container.innerHTML = `
    <div class="architecture-map">
      ${architectureNode('tokens', 'Token IDs', 'A | tiny | robot | learns')}
      ${flowArrow('computed sequence')}
      ${architectureNode('embedding', 'Embedding lookup', 'copy learned rows into X', 'computed', 'token embedding E')}
      ${flowArrow()}
      <div class="quartet-bracket ${cycleActive ? 'active' : ''}" data-target="cycle">
        <div class="quartet-title">
          <span>ONE 3:1 QUARTET</span>
          <strong>Repeated 8 times in Qwen3.5-4B</strong>
        </div>
        ${layerCard(0, 'delta')}
        ${flowArrow()}
        <div class="repeat-delta-bracket ${currentStep().architectureTarget === 'repeat-delta' ? 'active' : ''}" data-target="repeat-delta">
          <span>same GDN graph, new parameters</span>
          ${layerCard(1, 'delta')}
          ${flowArrow()}
          ${layerCard(2, 'delta')}
        </div>
        ${flowArrow()}
        ${layerCard(3, 'attention')}
      </div>
      ${flowArrow()}
      ${architectureNode('final-norm', 'Final RMSNorm', 'after all decoder layers')}
      ${flowArrow()}
      ${architectureNode('lm-head', 'Tied LM head', 'H x E-transpose', 'computed', 'same embedding E')}
      ${flowArrow()}
      ${architectureNode('next-token', 'Next-token distribution', 'softmax -> decode')}
    </div>
  `;

  window.requestAnimationFrame(() => {
    const target = container.querySelector(
      `[data-target="${currentStep().architectureTarget}"]`,
    );
    if (!target) return;
    const top =
      target.offsetTop - container.clientHeight / 2 + target.clientHeight / 2;
    container.scrollTo({
      top: Math.max(0, top),
      behavior: prefersReducedMotion ? 'auto' : 'smooth',
    });
  });
}

function renderOperation() {
  const pane = document.querySelector('#operation-pane');
  const step = currentStep();
  pane.innerHTML = `
    <div class="operation-header">
      <div>
        <p>${escapeHtml(step.eyebrow)}</p>
        <h2>${escapeHtml(step.title)}</h2>
      </div>
      <div class="step-chip">${state.stepIndex + 1} / ${QWEN_TUTORIAL.steps.length}</div>
    </div>
    <div class="operation-callouts">
      ${step.callouts
        .map(
          ([label, value]) => `
            <div><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>
          `,
        )
        .join('')}
    </div>
    ${
      currentStep().visual.kind === 'delta-repeat'
        ? `<div class="interaction-note summary-note">
            <i class="green-swatch"></i>
            <span><strong>Fast-forward summary.</strong> The matrices shown are real outputs from the teaching calculation; repeated cell-level explanations are intentionally omitted.</span>
          </div>`
        : `<div class="interaction-note">
            <i class="green-swatch"></i>
            <span><strong>Computed cells are interactive.</strong> Click a green output cell to trace exactly how it was produced.</span>
          </div>`
    }
    <div class="visual-stage" id="visual-stage"></div>
  `;
  renderVisualOnly();
}

function operationTabs(operations) {
  if (operations.length <= 1) return '';
  return `
    <div class="visual-tabs" role="tablist">
      ${operations
        .map(
          (operation, index) => `
            <button
              type="button"
              role="tab"
              aria-selected="${index === state.visualTab}"
              class="${index === state.visualTab ? 'active' : ''}"
              data-visual-tab="${index}"
            >${escapeHtml(operation.tabLabel)}</button>
          `,
        )
        .join('')}
    </div>
  `;
}

function bindVisualTabs(operations) {
  document.querySelectorAll('[data-visual-tab]').forEach((button) => {
    button.addEventListener('click', () => {
      state.visualTab = Number(button.dataset.visualTab);
      state.selectedCell = [
        ...operations[state.visualTab].defaultCell,
      ];
      state.activeTerm = 0;
      state.paused = prefersReducedMotion;
      renderVisualOnly();
    });
  });
}

function selectedMatmulTerms(operation) {
  const [rowIndex, colIndex] = state.selectedCell;
  return Array.from(
    { length: operation.left.values[0].length },
    (_, innerIndex) => {
      const leftValue = operation.left.values[rowIndex][innerIndex];
      const rightValue = operation.right.values[innerIndex][colIndex];
      return {
        innerIndex,
        leftValue,
        rightValue,
        product:
          leftValue * rightValue * (operation.termScale ?? 1),
      };
    },
  );
}

function matmulMarkup(operation, includeHeader = true) {
  if (!state.selectedCell) {
    state.selectedCell = [...operation.defaultCell];
  }
  const [rowIndex, colIndex] = state.selectedCell;
  const terms = selectedMatmulTerms(operation);
  const activeTerm = Math.min(state.activeTerm, terms.length - 1);
  const runningSum = terms
    .slice(0, activeTerm + 1)
    .reduce((sum, term) => sum + term.product, 0);
  const finalValue = operation.result.values[rowIndex][colIndex];
  const activeLeft = [[rowIndex, activeTerm]];
  const activeRight = [[activeTerm, colIndex]];
  const rightIsWeight = operation.right.role === 'weight';

  return `
    <div class="matmul-trace">
      ${
        includeHeader
          ? `
            <div class="visual-header">
              <div>
                <p>Traceable matrix multiplication</p>
                <h3>${escapeHtml(operation.title)}</h3>
                <code>${escapeHtml(operation.equation)}</code>
              </div>
              <button type="button" class="animation-toggle" id="animation-toggle">
                ${
                  state.paused &&
                  state.activeTerm >= operation.left.values[0].length - 1
                    ? 'Replay terms'
                    : state.paused
                      ? 'Play terms'
                      : 'Pause terms'
                }
              </button>
            </div>
            <p class="visual-caption">${escapeHtml(operation.caption)}</p>
          `
          : ''
      }
      <div class="operand-flow-legend">
        <span class="computed-flow-inline"><i></i>green computed row</span>
        <span class="${rightIsWeight ? 'weight-flow-inline' : 'computed-flow-inline'}">
          <i></i>${rightIsWeight ? 'blue learned column' : 'green computed column'}
        </span>
        <b>meet in one green output cell</b>
      </div>
      <div class="matmul-equation">
        ${matrixGrid(operation.left, {
          highlightRow: rowIndex,
          activeCells: activeLeft,
        })}
        <div class="math-operator computed-connector" aria-hidden="true">
          <span>x</span><i></i>
        </div>
        ${matrixGrid(operation.right, {
          highlightCol: colIndex,
          activeCells: activeRight,
        })}
        <div class="math-operator ${rightIsWeight ? 'weight-connector' : 'computed-connector'}" aria-hidden="true">
          <span>=</span><i></i>
        </div>
        ${matrixGrid(operation.result, {
          selectedCell: state.selectedCell,
          clickable: true,
        })}
      </div>
      <div class="dot-product-inspector">
        <div class="inspector-topline">
          <span>Selected computed cell</span>
          <code>${escapeHtml(`${operation.result.symbol}[${rowIndex},${colIndex}] = ${formatCell(finalValue)}`)}</code>
        </div>
        <div class="term-strip">
          ${terms
            .map(
              (term, index) => `
                <button
                  type="button"
                  class="term-chip ${index === activeTerm ? 'active' : ''}"
                  data-term-index="${index}"
                >
                  <small>k=${term.innerIndex}</small>
                  <span>
                    ${formatCell(term.leftValue)} x ${formatCell(term.rightValue)}
                    ${(operation.termScale ?? 1) !== 1 ? ` x ${formatCell(operation.termScale)}` : ''}
                  </span>
                  <strong>${formatCell(term.product)}</strong>
                </button>
              `,
            )
            .join('')}
        </div>
        <div class="running-sum">
          <span>running sum through term ${activeTerm + 1}</span>
          <strong>${formatCell(runningSum)}</strong>
          <i></i>
          <span>final cell</span>
          <strong>${formatCell(finalValue)}</strong>
        </div>
      </div>
      ${
        operation.derived
          ? `
            <div class="derived-result">
              <div class="derived-arrow"><i></i><span>element-wise transform</span></div>
              <div class="derived-copy">
                <code>${escapeHtml(operation.derived.formula)}</code>
                <p>${escapeHtml(operation.derived.caption)}</p>
              </div>
              ${matrixGrid(operation.derived.matrix, { compact: true })}
            </div>
          `
          : ''
      }
    </div>
  `;
}

function bindMatmul(operation) {
  document.querySelectorAll('[data-output-cell]').forEach((cell) => {
    cell.addEventListener('click', () => {
      state.selectedCell = [
        Number(cell.dataset.row),
        Number(cell.dataset.col),
      ];
      state.activeTerm = 0;
      state.paused = prefersReducedMotion;
      renderVisualOnly();
    });
  });
  document.querySelectorAll('[data-term-index]').forEach((term) => {
    term.addEventListener('click', () => {
      state.activeTerm = Number(term.dataset.termIndex);
      state.paused = true;
      renderVisualOnly();
    });
  });
  document.querySelector('#animation-toggle')?.addEventListener('click', () => {
    if (
      state.paused &&
      state.activeTerm >= operation.left.values[0].length - 1
    ) {
      state.activeTerm = 0;
      state.paused = false;
    } else {
      state.paused = !state.paused;
    }
    renderVisualOnly();
  });
  scheduleTerms(operation.left.values[0].length);
}

function renderMatmul(visual) {
  const operation = visual.operations[state.visualTab];
  const stage = document.querySelector('#visual-stage');
  stage.innerHTML = `
    ${operationTabs(visual.operations)}
    ${matmulMarkup(operation)}
  `;
  bindVisualTabs(visual.operations);
  bindMatmul(operation);
}

function renderTransform(visual) {
  const operation = visual.operations[state.visualTab];
  if (!state.selectedCell) {
    state.selectedCell = [...operation.defaultCell];
  }
  const [rowIndex, colIndex] = state.selectedCell;
  const detail = operation.details[rowIndex][colIndex];
  const stage = document.querySelector('#visual-stage');
  stage.innerHTML = `
    ${operationTabs(visual.operations)}
    <div class="transform-visual">
      <div class="visual-header">
        <div>
          <p>Row-local / element-wise transform</p>
          <h3>${escapeHtml(operation.title)}</h3>
          <code>${escapeHtml(operation.formula)}</code>
        </div>
      </div>
      <p class="visual-caption">${escapeHtml(operation.caption)}</p>
      <div class="transform-equation">
        ${matrixGrid(operation.input, { highlightRow: rowIndex })}
        <div class="green-forward-arrow"><i></i><span>compute</span></div>
        ${matrixGrid(operation.output, {
          selectedCell: state.selectedCell,
          clickable: true,
        })}
      </div>
      <div class="cell-detail">
        <span>Selected green cell</span>
        <code>${escapeHtml(detail)}</code>
      </div>
      ${
        operation.auxiliaries?.length
          ? `
            <div class="auxiliary-matrices">
              <p>Blue learned values used by this transform</p>
              ${operation.auxiliaries
                .map((matrix) => matrixGrid(matrix, { compact: true }))
                .join('')}
            </div>
          `
          : ''
      }
    </div>
  `;
  bindVisualTabs(visual.operations);
  document.querySelectorAll('[data-output-cell]').forEach((cell) => {
    cell.addEventListener('click', () => {
      state.selectedCell = [
        Number(cell.dataset.row),
        Number(cell.dataset.col),
      ];
      renderVisualOnly();
    });
  });
}

function renderElementwise(visual) {
  if (!state.selectedCell) state.selectedCell = [...visual.defaultCell];
  const [rowIndex, colIndex] = state.selectedCell;
  const detail = visual.details[rowIndex][colIndex];
  const stage = document.querySelector('#visual-stage');
  stage.innerHTML = `
    <div class="elementwise-visual">
      <div class="visual-header">
        <div>
          <p>Coordinate-wise operation</p>
          <h3>${escapeHtml(visual.title)}</h3>
          <code>${escapeHtml(visual.formula)}</code>
        </div>
      </div>
      <p class="visual-caption">${escapeHtml(visual.caption)}</p>
      <div class="matmul-equation elementwise-equation">
        ${matrixGrid(visual.left, {
          activeCells: [[rowIndex, colIndex]],
        })}
        <div class="math-operator computed-connector"><span>${visual.operator === 'add' ? '+' : 'x'}</span><i></i></div>
        ${matrixGrid(visual.right, {
          activeCells: [[rowIndex, colIndex]],
        })}
        <div class="math-operator computed-connector"><span>=</span><i></i></div>
        ${matrixGrid(visual.result, {
          selectedCell: state.selectedCell,
          clickable: true,
        })}
      </div>
      <div class="cell-detail">
        <span>Selected green cell</span>
        <code>${escapeHtml(detail)}</code>
      </div>
    </div>
  `;
  document.querySelectorAll('[data-output-cell]').forEach((cell) => {
    cell.addEventListener('click', () => {
      state.selectedCell = [
        Number(cell.dataset.row),
        Number(cell.dataset.col),
      ];
      renderVisualOnly();
    });
  });
}

function renderTokens(visual) {
  const stage = document.querySelector('#visual-stage');
  stage.innerHTML = `
    <div class="token-visual">
      <div class="visual-header">
        <div>
          <p>Tokenizer output</p>
          <h3>Strings stop here; integer IDs enter the model</h3>
          <code>["A", "tiny", "robot", "learns"] -> [1, 2, 3, 4]</code>
        </div>
      </div>
      <div class="token-flow">
        ${visual.tokens
          .map(
            (token, index) => `
              <div class="token-column">
                <div class="text-token">${escapeHtml(token)}</div>
                <div class="green-forward-arrow compact"><i></i></div>
                <div class="token-id"><span>id</span><strong>${visual.tokenIds[index]}</strong></div>
              </div>
            `,
          )
          .join('')}
      </div>
      <div class="vocabulary-strip">
        <span>Toy vocabulary</span>
        ${visual.vocabulary
          .map(
            (token, index) =>
              `<code class="${visual.tokenIds.includes(index) ? 'used' : ''}">${index}:${escapeHtml(token)}</code>`,
          )
          .join('')}
      </div>
    </div>
  `;
}

function renderLookup(visual) {
  const selected = state.lookupToken;
  const tokenId = visual.tokenIds[selected];
  const stage = document.querySelector('#visual-stage');
  const embeddingActive = visual.embeddingTable.values[tokenId].map(
    (_, colIndex) => [tokenId, colIndex],
  );
  const outputActive = visual.result.values[selected].map(
    (_, colIndex) => [selected, colIndex],
  );

  stage.innerHTML = `
    <div class="lookup-visual">
      <div class="visual-header">
        <div>
          <p>Learned row lookup</p>
          <h3>Token "${escapeHtml(visual.tokens[selected])}" selects embedding row ${tokenId}</h3>
          <code>X[${selected},:] = E[${tokenId},:]</code>
        </div>
      </div>
      <div class="lookup-token-tabs">
        ${visual.tokens
          .map(
            (token, index) => `
              <button type="button" data-lookup-token="${index}" class="${index === selected ? 'active' : ''}">
                <span>${escapeHtml(token)}</span><strong>id ${visual.tokenIds[index]}</strong>
              </button>
            `,
          )
          .join('')}
      </div>
      <div class="lookup-equation">
        ${matrixGrid(visual.embeddingTable, {
          activeCells: embeddingActive,
        })}
        <div class="blue-to-green-arrow">
          <span>copy blue row</span><i></i><strong>computed values</strong>
        </div>
        ${matrixGrid(visual.result, {
          activeCells: outputActive,
        })}
      </div>
    </div>
  `;
  document.querySelectorAll('[data-lookup-token]').forEach((button) => {
    button.addEventListener('click', () => {
      state.lookupToken = Number(button.dataset.lookupToken);
      renderVisualOnly();
    });
  });
}

function configRows(config) {
  return [
    ['hidden width', config.hiddenSize.toLocaleString()],
    ['decoder layers', config.layers.toLocaleString()],
    ['layer pattern', config.pattern],
    ['DeltaNet heads', config.deltaHeads],
    ['attention heads', config.attentionHeads],
    ['FFN width', config.ffnSize.toLocaleString()],
    ['vocabulary', config.vocabulary.toLocaleString()],
  ];
}

function renderOverview() {
  const stage = document.querySelector('#visual-stage');
  const rows = configRows(MINI_CONFIG)
    .map(
      ([label, miniValue], index) => `
        <div class="comparison-label">${escapeHtml(label)}</div>
        <div>${escapeHtml(miniValue)}</div>
        <div>${escapeHtml(configRows(REAL_CONFIG)[index][1])}</div>
      `,
    )
    .join('');
  stage.innerHTML = `
    <div class="overview-visual">
      <div class="visual-header">
        <div>
          <p>Same graph, smaller tensors</p>
          <h3>The mini model is a microscope, not a checkpoint</h3>
          <code>1 teaching quartet = layers 0, 1, 2, 3</code>
        </div>
      </div>
      <div class="comparison-table">
        <div></div><strong>Teaching scale</strong><strong>Qwen3.5-4B</strong>
        ${rows}
      </div>
      <div class="mini-quartet">
        <div class="mini-layer delta"><span>0</span><strong>Gated DeltaNet</strong><small>+ dense SwiGLU FFN</small></div>
        <div class="vertical-flow computed-flow"><i></i></div>
        <div class="mini-layer delta"><span>1</span><strong>Gated DeltaNet</strong><small>+ dense SwiGLU FFN</small></div>
        <div class="vertical-flow computed-flow"><i></i></div>
        <div class="mini-layer delta"><span>2</span><strong>Gated DeltaNet</strong><small>+ dense SwiGLU FFN</small></div>
        <div class="vertical-flow computed-flow"><i></i></div>
        <div class="mini-layer attention"><span>3</span><strong>Gated full attention</strong><small>+ dense SwiGLU FFN</small></div>
      </div>
    </div>
  `;
}

function renderConvolution(visual) {
  if (!state.selectedCell) state.selectedCell = [...visual.defaultCell];
  const [timeIndex, channelIndex] = state.selectedCell;
  const trace = visual.traces[timeIndex][channelIndex];
  const activeTerm = Math.min(state.activeTerm, trace.terms.length - 1);
  const term = trace.terms[activeTerm];
  const inputActive =
    term.inputTime === null ? [] : [[term.inputTime, channelIndex]];
  const stage = document.querySelector('#visual-stage');
  stage.innerHTML = `
    <div class="convolution-visual">
      <div class="visual-header">
        <div>
          <p>Local causal sequence mixing</p>
          <h3>One four-tap filter per Q/K/V channel</h3>
          <code>out[t,c] = SiLU(sum_r input[t-r,c] x kernel[c,r])</code>
        </div>
        <button type="button" class="animation-toggle" id="animation-toggle">
          ${
            state.paused &&
            state.activeTerm >= visual.kernels.values[0].length - 1
              ? 'Replay taps'
              : state.paused
                ? 'Play taps'
                : 'Pause taps'
          }
        </button>
      </div>
      <div class="operand-flow-legend">
        <span class="computed-flow-inline"><i></i>green projected channel</span>
        <span class="weight-flow-inline"><i></i>blue convolution taps</span>
        <b>future positions are never used</b>
      </div>
      <div class="matmul-equation">
        ${matrixGrid(visual.input, {
          activeCells: inputActive,
          highlightCol: channelIndex,
        })}
        <div class="math-operator computed-connector"><span>*</span><i></i></div>
        ${matrixGrid(visual.kernels, {
          activeCells: [[channelIndex, activeTerm]],
          highlightRow: channelIndex,
        })}
        <div class="math-operator weight-connector"><span>=</span><i></i></div>
        ${matrixGrid(visual.output, {
          selectedCell: state.selectedCell,
          clickable: true,
        })}
      </div>
      <div class="dot-product-inspector">
        <div class="inspector-topline">
          <span>Selected convolution cell</span>
          <code>${escapeHtml(`${visual.output.symbol}[${timeIndex},${channelIndex}] = ${formatCell(trace.activated)}`)}</code>
        </div>
        <div class="term-strip">
          ${trace.terms
            .map(
              (item, index) => `
                <button
                  type="button"
                  class="term-chip ${index === activeTerm ? 'active' : ''}"
                  data-term-index="${index}"
                >
                  <small>${item.inputTime === null ? 'left pad' : `t${item.inputTime}`}</small>
                  <span>${formatCell(item.inputValue)} x ${formatCell(item.kernelValue)}</span>
                  <strong>${formatCell(item.product)}</strong>
                </button>
              `,
            )
            .join('')}
        </div>
        <div class="running-sum">
          <span>four-tap sum</span><strong>${formatCell(trace.sum)}</strong>
          <i></i><span>SiLU</span><strong>${formatCell(trace.activated)}</strong>
        </div>
      </div>
    </div>
  `;
  bindConvolution(visual);
}

function bindConvolution(visual) {
  document.querySelectorAll('[data-output-cell]').forEach((cell) => {
    cell.addEventListener('click', () => {
      state.selectedCell = [
        Number(cell.dataset.row),
        Number(cell.dataset.col),
      ];
      state.activeTerm = 0;
      state.paused = prefersReducedMotion;
      renderVisualOnly();
    });
  });
  document.querySelectorAll('[data-term-index]').forEach((button) => {
    button.addEventListener('click', () => {
      state.activeTerm = Number(button.dataset.termIndex);
      state.paused = true;
      renderVisualOnly();
    });
  });
  document.querySelector('#animation-toggle')?.addEventListener('click', () => {
    if (
      state.paused &&
      state.activeTerm >= visual.kernels.values[0].length - 1
    ) {
      state.activeTerm = 0;
      state.paused = false;
    } else {
      state.paused = !state.paused;
    }
    renderVisualOnly();
  });
  scheduleTerms(visual.kernels.values[0].length);
}

function deltaOperation(trace, phase) {
  const valueLabels = trace.v.colLabels;
  if (phase === 'write') {
    return makeMatmulOperation({
      id: `delta-write-${trace.tokenIndex}-${trace.headIndex}`,
      tabLabel: 'Write',
      title: 'Rank-one state write',
      equation: 'write = k_t^T x delta_t',
      caption:
        'Every state cell receives one key component times one correction component.',
      left: matrixData(
        'transposed key',
        'k_t^T',
        transpose(trace.k.values),
        'activation',
        ['k0', 'k1'],
        ['token'],
      ),
      right: trace.correction,
      resultName: 'rank-one write',
      resultSymbol: 'write',
      resultRowLabels: ['k0', 'k1'],
      resultColLabels: valueLabels,
      resultValues: trace.write.values,
      defaultCell: [0, 0],
    });
  }
  if (phase === 'read') {
    return makeMatmulOperation({
      id: `delta-read-${trace.tokenIndex}-${trace.headIndex}`,
      tabLabel: 'Read',
      title: 'Query reads the updated state',
      equation: 'o_t = q_t x S_t',
      caption:
        'The query combines state rows into the value-head output.',
      left: trace.q,
      right: trace.stateAfter,
      resultName: 'head output',
      resultSymbol: 'o_t',
      resultRowLabels: [`t${trace.tokenIndex}`],
      resultColLabels: valueLabels,
      resultValues: trace.read.values,
      defaultCell: [0, 0],
    });
  }
  return makeMatmulOperation({
    id: `delta-predict-${trace.tokenIndex}-${trace.headIndex}`,
    tabLabel: 'Predict',
    title: 'Key asks what the state already predicts',
    equation: 'memory = k_t x (exp(g_t) S_(t-1))',
    caption:
      'The value target will write only the beta-scaled prediction error.',
    left: trace.k,
    right: trace.stateDecayed,
    resultName: 'state prediction',
    resultSymbol: 'memory',
    resultRowLabels: [`t${trace.tokenIndex}`],
    resultColLabels: valueLabels,
    resultValues: trace.memory.values,
    defaultCell: [0, 0],
  });
}

function deltaNonMatmulMarkup(trace, phase) {
  let left;
  let middle;
  let result;
  let operator;
  let formula;
  let detail;

  if (phase === 'decay') {
    left = trace.stateBefore;
    middle = matrixData(
      'retention factor',
      'exp(g_t)',
      [[trace.decay]],
      'gate',
      ['head'],
      ['scalar'],
    );
    result = trace.stateDecayed;
    operator = 'x';
    formula = 'S_tilde[r,c] = S_(t-1)[r,c] x exp(g_t)';
  } else if (phase === 'correct') {
    left = trace.v;
    middle = trace.memory;
    result = trace.correction;
    operator = '- then x beta';
    formula = 'delta_t[c] = (v_t[c] - memory[c]) x beta_t';
  } else {
    left = trace.stateDecayed;
    middle = trace.write;
    result = trace.stateAfter;
    operator = '+';
    formula = 'S_t[r,c] = S_tilde[r,c] + write[r,c]';
  }

  if (!state.selectedCell) {
    state.selectedCell = [0, 0];
  }
  const [rowIndex, colIndex] = state.selectedCell;
  if (phase === 'decay') {
    detail =
      `${formatCell(left.values[rowIndex][colIndex])} x ${formatCell(trace.decay)} = ` +
      `${formatCell(result.values[rowIndex][colIndex])}`;
  } else if (phase === 'correct') {
    detail =
      `(${formatCell(left.values[rowIndex][colIndex])} - ${formatCell(middle.values[rowIndex][colIndex])}) ` +
      `x ${formatCell(trace.beta)} = ${formatCell(result.values[rowIndex][colIndex])}`;
  } else {
    detail =
      `${formatCell(left.values[rowIndex][colIndex])} + ${formatCell(middle.values[rowIndex][colIndex])} = ` +
      `${formatCell(result.values[rowIndex][colIndex])}`;
  }

  return `
    <div class="elementwise-visual delta-elementwise">
      <div class="visual-header">
        <div>
          <p>Sequential DeltaNet state action</p>
          <h3>${escapeHtml(
            phase === 'decay'
              ? 'Apply retention to the previous state'
              : phase === 'correct'
                ? 'Compute the beta-scaled value error'
                : 'Commit the rank-one write to state',
          )}</h3>
          <code>${escapeHtml(formula)}</code>
        </div>
      </div>
      <div class="matmul-equation elementwise-equation">
        ${matrixGrid(left, {
          activeCells: [[rowIndex, colIndex]],
        })}
        <div class="math-operator computed-connector"><span>${escapeHtml(operator)}</span><i></i></div>
        ${matrixGrid(middle, {
          activeCells:
            phase === 'decay' ? [[0, 0]] : [[rowIndex, colIndex]],
        })}
        ${
          phase === 'correct'
            ? `<div class="delta-beta-chip">beta = ${formatCell(trace.beta)}</div>`
            : ''
        }
        <div class="math-operator computed-connector"><span>=</span><i></i></div>
        ${matrixGrid(result, {
          selectedCell: state.selectedCell,
          clickable: true,
        })}
      </div>
      <div class="cell-detail">
        <span>Selected green cell</span>
        <code>${escapeHtml(detail)}</code>
      </div>
    </div>
  `;
}

function deltaChainMarkup(visual) {
  const traces = visual.traces
    .filter((item) => item.headIndex === state.deltaHead)
    .sort((left, right) => left.tokenIndex - right.tokenIndex);
  const initialState = matrixData(
    'initial state before t0',
    'S_-1',
    traces[0].stateBefore.values,
    'state',
    traces[0].stateBefore.rowLabels,
    traces[0].stateBefore.colLabels,
  );

  return `
    <div class="delta-chain-visual">
      <div class="visual-header">
        <div>
          <p>One state, advanced left to right</p>
          <h3>Value head ${state.deltaHead} keeps a single recurrent memory chain</h3>
          <code>S_-1 -> S_0 -> S_1 -> S_2 -> S_3</code>
        </div>
      </div>
      <p class="visual-caption">
        The state is not reset for each token. Every S_t below is the input state for the next token.
      </p>
      <div class="delta-chain">
        <div class="state-checkpoint initial">
          ${matrixGrid(initialState, { compact: true })}
          <small>before the sequence</small>
        </div>
        ${traces
          .map((item) => {
            const stateAfter = matrixData(
              `state after t${item.tokenIndex}`,
              `S_${item.tokenIndex}`,
              item.stateAfter.values,
              'state',
              item.stateAfter.rowLabels,
              item.stateAfter.colLabels,
            );
            const active = item.tokenIndex === state.deltaToken;
            return `
              <div class="chain-transition ${active ? 'active' : ''}">
                <div class="chain-arrow">
                  <span>process t${item.tokenIndex} ${escapeHtml(visual.tokens[item.tokenIndex])}</span>
                  <i></i>
                  <small>produces o_${item.tokenIndex} = [${item.read.values[0].map(formatCell).join(', ')}]</small>
                </div>
                <div class="state-checkpoint">
                  ${matrixGrid(stateAfter, { compact: true })}
                  <small>carried into ${item.tokenIndex + 1 < traces.length ? `t${item.tokenIndex + 1}` : 'the next generated token'}</small>
                </div>
              </div>
            `;
          })
          .join('')}
      </div>
      <div class="chain-clarifier">
        <strong>Independent across heads, sequential across tokens.</strong>
        Head 0 has this chain and head 1 has a separate chain. Within either chain, token t cannot be evaluated from a fresh state.
      </div>
    </div>
  `;
}

function deltaAssemblyMarkup(visual) {
  if (!state.selectedCell) {
    state.selectedCell = [3, 2];
  }
  const [tokenIndex, coreCol] = state.selectedCell;
  const headWidth = visual.headOutputs[0].values[0].length;
  const sourceHead = Math.floor(coreCol / headWidth);
  const sourceCol = coreCol % headWidth;
  const source = visual.headOutputs[sourceHead];
  const sourceValue = source.values[tokenIndex][sourceCol];
  const resultValue = visual.coreOutput.values[tokenIndex][coreCol];

  return `
    <div class="delta-assembly-visual">
      <div class="visual-header">
        <div>
          <p>Collect recurrent reads</p>
          <h3>Stack o_t by time, then concatenate value heads</h3>
          <code>O_core[t,:] = concat(o_t^(head 0), o_t^(head 1))</code>
        </div>
      </div>
      <p class="visual-caption">
        Each row below came from the read step q_t S_t at that token. Concatenation changes layout only; it performs no new arithmetic.
      </p>
      <div class="assembly-equation">
        <div class="head-output-stack">
          ${visual.headOutputs
            .map((matrix, headIndex) =>
              matrixGrid(matrix, {
                activeCells:
                  headIndex === sourceHead
                    ? [[tokenIndex, sourceCol]]
                    : [],
                compact: true,
              }),
            )
            .join('')}
        </div>
        <div class="green-forward-arrow"><i></i><span>concatenate columns</span></div>
        ${matrixGrid(visual.coreOutput, {
          selectedCell: state.selectedCell,
          clickable: true,
        })}
      </div>
      <div class="cell-detail">
        <span>Selected O_core cell</span>
        <code>
          ${escapeHtml(
            `O_core[${tokenIndex},${coreCol}] copies ${source.symbol}[${tokenIndex},${sourceCol}] = ${formatCell(sourceValue)} -> ${formatCell(resultValue)}`,
          )}
        </code>
      </div>
      <div class="chain-clarifier">
        <strong>This is the missing o_t -> O_core bridge.</strong>
        Rows are token positions; columns v0/v1 come from head 0 and v2/v3 come from head 1. O_core now flows directly into per-head RMSNorm and SiLU(z).
      </div>
    </div>
  `;
}

function renderDelta(visual) {
  const trace = visual.traces.find(
    (item) =>
      item.tokenIndex === state.deltaToken &&
      item.headIndex === state.deltaHead,
  );
  const phase = visual.phase;
  const phaseOrder = ['chain', 'decay', 'predict', 'correct', 'write', 'update', 'read', 'assemble'];
  const phaseIndex = phaseOrder.indexOf(phase);
  const isMatmul = ['predict', 'write', 'read'].includes(phase);
  const operation = isMatmul ? deltaOperation(trace, phase) : null;
  if (!state.selectedCell && operation) {
    state.selectedCell = [...operation.defaultCell];
  }
  const operationMarkup =
    phase === 'chain'
      ? deltaChainMarkup(visual)
      : phase === 'assemble'
        ? deltaAssemblyMarkup(visual)
        : isMatmul
          ? matmulMarkup(operation)
          : deltaNonMatmulMarkup(trace, phase);
  const stage = document.querySelector('#visual-stage');
  stage.innerHTML = `
    <div class="delta-visual">
      <div class="delta-phase-banner">
        <span>Current scan action</span>
        <strong>${escapeHtml(
          {
            chain: '1. State chain overview',
            decay: '2. Decay state',
            predict: '3. Predict kS',
            correct: '4. Correct value error',
            write: '5. Write k^T delta',
            update: '6. Update state',
            read: '7. Read qS',
            assemble: '8. Assemble O_core',
          }[phase],
        )}</strong>
        <small>Continue advances to the next dependent action.</small>
      </div>
      ${
        phase !== 'assemble'
          ? `<div class="delta-controls">
        <div>
          <span>Inspect recurrent checkpoint</span>
          <div class="delta-time-sequence">
            ${visual.tokens
              .map(
                (token, index) => `
                  ${index > 0 ? '<i aria-hidden="true"></i>' : ''}
                  <button type="button" data-delta-token="${index}" class="${index === state.deltaToken ? 'active' : ''}">
                    t${index} ${escapeHtml(token)}
                  </button>
                `,
              )
              .join('')}
          </div>
          <small>One state chain; these buttons only choose where to inspect it.</small>
        </div>
        <div>
          <span>Independent value-head memory</span>
          ${Array.from(
            { length: visual.heads },
            (_, index) => `
              <button type="button" data-delta-head="${index}" class="${index === state.deltaHead ? 'active' : ''}">
                head ${index}
              </button>
            `,
          ).join('')}
        </div>
      </div>`
          : ''
      }
      ${
        !['chain', 'assemble'].includes(phase)
          ? `<div class="delta-state-strip">
        ${matrixGrid(trace.stateBefore, { compact: true })}
        <div class="state-arrow">
          <span>decay x ${formatCell(trace.decay)}</span><i></i>
        </div>
        ${matrixGrid(trace.stateDecayed, { compact: true })}
        ${
          phaseIndex >= 5
            ? `
              <div class="state-arrow">
                <span>+ rank-one write</span><i></i>
              </div>
              ${matrixGrid(trace.stateAfter, { compact: true })}
            `
            : ''
        }
      </div>
      <div class="delta-scalars">
        <span><b>retention exp(g)</b>${formatCell(trace.decay)}</span>
        <span><b>write rate beta</b>${formatCell(trace.beta)}</span>
        ${
          phaseIndex >= 3
            ? `<span><b>correction</b>${trace.correction.values[0].map(formatCell).join(', ')}</span>`
            : ''
        }
      </div>`
          : ''
      }
      ${operationMarkup}
    </div>
  `;

  document.querySelectorAll('[data-delta-token]').forEach((button) => {
    button.addEventListener('click', () => {
      state.deltaToken = Number(button.dataset.deltaToken);
      state.selectedCell = null;
      state.activeTerm = 0;
      state.paused = prefersReducedMotion;
      renderVisualOnly();
    });
  });
  document.querySelectorAll('[data-delta-head]').forEach((button) => {
    button.addEventListener('click', () => {
      state.deltaHead = Number(button.dataset.deltaHead);
      state.selectedCell = null;
      state.activeTerm = 0;
      state.paused = prefersReducedMotion;
      renderVisualOnly();
    });
  });
  if (isMatmul) {
    bindMatmul(operation);
  } else if (phase !== 'chain') {
    document.querySelectorAll('[data-output-cell]').forEach((cell) => {
      cell.addEventListener('click', () => {
        state.selectedCell = [
          Number(cell.dataset.row),
          Number(cell.dataset.col),
        ];
        renderVisualOnly();
      });
    });
  }
}

function renderGqa(visual) {
  const stage = document.querySelector('#visual-stage');
  stage.innerHTML = `
    <div class="gqa-visual">
      <div class="visual-header">
        <div>
          <p>Grouped-query head routing</p>
          <h3>${visual.qHeads} query heads point to one shared K/V pair</h3>
          <code>${Array.from({ length: visual.qHeads }, (_, index) => `Q${index}`).join(', ')} -> K0 / V0</code>
        </div>
      </div>
      <div class="gqa-map">
        <div class="gqa-query-stack">
          ${visual.q
            .map(
              (matrix, index) => `
                <div class="head-card">
                  <span>QUERY HEAD ${index}</span>
                  ${matrixGrid(matrix, { compact: true })}
                </div>
              `,
            )
            .join('')}
        </div>
        <div class="gqa-arrows" aria-hidden="true">
          ${Array.from({ length: visual.qHeads }, () => '<i></i>').join('')}
          <strong>share ${visual.qHeads}:1</strong>
        </div>
        <div class="gqa-shared-stack">
          <div class="head-card shared">
            <span>SHARED KEY HEAD</span>
            ${matrixGrid(visual.k[0], { compact: true })}
          </div>
          <div class="head-card shared">
            <span>SHARED VALUE HEAD</span>
            ${matrixGrid(visual.v[0], { compact: true })}
          </div>
        </div>
      </div>
      <div class="gqa-note">
        <span>Mini: ${MINI_CONFIG.attentionQueryHeads} Q -> ${MINI_CONFIG.attentionKvHeads} K/V</span>
        <i></i>
        <span>Qwen3.5-4B: ${REAL_CONFIG.attentionQueryHeads} Q -> ${REAL_CONFIG.attentionKvHeads} K/V</span>
      </div>
    </div>
  `;
}

function renderDeltaRepeat(visual) {
  const stage = document.querySelector('#visual-stage');
  stage.innerHTML = `
    <div class="delta-repeat-visual">
      <div class="visual-header">
        <div>
          <p>Repeated decoder layers</p>
          <h3>Same graph twice; new learned weights and new recurrent states</h3>
          <code>X_after_L0 -> layer 1 -> X_after_L1 -> layer 2 -> X_after_L2</code>
        </div>
      </div>
      <p class="visual-caption">
        The teaching model still computes every operation in both layers. This view collapses only the repeated explanation.
      </p>
      <div class="repeat-delta-flow">
        ${matrixGrid(visual.stages[0], { compact: true })}
        ${visual.transitions
          .map(
            (transition, index) => `
              <div class="repeat-layer-transition">
                <span>DECODER LAYER ${transition.layer}</span>
                <strong>${escapeHtml(transition.label)}</strong>
                <small>${escapeHtml(transition.detail)}</small>
                <i></i>
              </div>
              ${matrixGrid(visual.stages[index + 1], { compact: true })}
            `,
          )
          .join('')}
      </div>
      <div class="repeat-facts">
        <div><span>Reused</span><strong>operation graph and tensor shapes</strong></div>
        <div><span>Not reused</span><strong>projection weights, norm scales, convolution kernels</strong></div>
        <div><span>State rule</span><strong>each layer owns two fresh value-head state chains</strong></div>
      </div>
      <div class="chain-clarifier">
        <strong>No model computation was skipped.</strong>
        The next detailed chapter starts with the real output of layer 2 as the input to layer 3 full attention.
      </div>
    </div>
  `;
}

function renderSoftmax(visual) {
  const rowIndex = state.softmaxRow;
  const headIndex = visual.headIndex;
  const probabilities = visual.probabilities.values[rowIndex];
  const isMaskStep = visual.phase === 'mask';
  const stage = document.querySelector('#visual-stage');
  stage.innerHTML = `
    <div class="softmax-visual">
      <div class="visual-header">
        <div>
          <p>Causal normalization</p>
          <h3>${
            isMaskStep
              ? `Mask future scores for query head ${headIndex}`
              : `Normalize surviving scores for query head ${headIndex}`
          }</h3>
          <code>${
            isMaskStep
              ? `Masked${headIndex}[t${rowIndex},j] = j > ${rowIndex} ? -inf : Scores${headIndex}[t${rowIndex},j]`
              : `P${headIndex}[t${rowIndex},:] = softmax(Masked${headIndex}[t${rowIndex},:])`
          }</code>
        </div>
      </div>
      <div class="row-tabs">
        ${visual.scores.values
          .map(
            (_, index) => `
              <button type="button" data-softmax-row="${index}" class="${index === rowIndex ? 'active' : ''}">
                query t${index}
              </button>
            `,
          )
          .join('')}
      </div>
      <div class="softmax-flow">
        ${
          isMaskStep
            ? `
              ${matrixGrid(visual.scores, { highlightRow: rowIndex })}
              <div class="green-forward-arrow"><i></i><span>causal mask</span></div>
              ${matrixGrid(visual.maskedScores, { highlightRow: rowIndex })}
            `
            : `
              ${matrixGrid(visual.maskedScores, { highlightRow: rowIndex })}
              <div class="green-forward-arrow"><i></i><span>row softmax</span></div>
              ${matrixGrid(visual.probabilities, { highlightRow: rowIndex })}
            `
        }
      </div>
      ${
        isMaskStep
          ? `
            <div class="mask-explanation">
              Head ${headIndex} entries after key t${rowIndex} are now -inf. The next Continue step turns them into zero probability.
            </div>
          `
          : `
            <div class="probability-bars">
              ${probabilities
                .map(
                  (value, index) => `
                    <div>
                      <span>key t${index}</span>
                      <i style="width:${value * 100}%"></i>
                      <strong>${(value * 100).toFixed(1)}%</strong>
                    </div>
                  `,
                )
                .join('')}
            </div>
            <div class="row-sum">Row sum: <strong>${probabilities.reduce((sum, value) => sum + value, 0).toFixed(3)}</strong></div>
          `
      }
    </div>
  `;
  document.querySelectorAll('[data-softmax-row]').forEach((button) => {
    button.addEventListener('click', () => {
      state.softmaxRow = Number(button.dataset.softmaxRow);
      renderVisualOnly();
    });
  });
}

function renderPrediction(visual) {
  const lastRow = visual.probabilities.values.length - 1;
  const probabilities = visual.probabilities.values[lastRow];
  const logits = visual.logits.values[lastRow];
  const stage = document.querySelector('#visual-stage');
  stage.innerHTML = `
    <div class="prediction-visual">
      <div class="visual-header">
        <div>
          <p>Autoregressive decode</p>
          <h3>Last-row logits become a vocabulary distribution</h3>
          <code>argmax(softmax(logits[t3,:])) = ${escapeHtml(visual.nextToken)}</code>
        </div>
      </div>
      <div class="prediction-bars">
        ${visual.tokens
          .map(
            (token, index) => `
              <div class="${index === visual.nextTokenId ? 'winner' : ''}">
                <span>${escapeHtml(token)}</span>
                <small>logit ${formatCell(logits[index])}</small>
                <i style="width:${probabilities[index] * 100}%"></i>
                <strong>${(probabilities[index] * 100).toFixed(1)}%</strong>
              </div>
            `,
          )
          .join('')}
      </div>
      <div class="next-token-card">
        <span>Greedy next token</span>
        <strong>${escapeHtml(visual.nextToken)}</strong>
        <div class="vertical-flow computed-flow"><i></i></div>
        <small>Append it, then run the model again for the following token.</small>
      </div>
      <p class="toy-warning">
        The weights on this page are deterministic teaching values, not Qwen checkpoint weights. The tied embedding row for "fast" is intentionally designed to win this example; the arithmetic and decoding path are real.
      </p>
    </div>
  `;
}

function renderVisualOnly() {
  window.clearTimeout(termTimer);
  termTimer = null;
  const visual = currentStep().visual;
  if (visual.kind === 'tokens') return renderTokens(visual);
  if (visual.kind === 'lookup') return renderLookup(visual);
  if (visual.kind === 'overview') return renderOverview();
  if (visual.kind === 'matmul') return renderMatmul(visual);
  if (visual.kind === 'transform') return renderTransform(visual);
  if (visual.kind === 'elementwise') return renderElementwise(visual);
  if (visual.kind === 'convolution') return renderConvolution(visual);
  if (visual.kind === 'delta') return renderDelta(visual);
  if (visual.kind === 'delta-repeat') return renderDeltaRepeat(visual);
  if (visual.kind === 'gqa') return renderGqa(visual);
  if (visual.kind === 'softmax') return renderSoftmax(visual);
  if (visual.kind === 'prediction') return renderPrediction(visual);
}

function scheduleTerms(termCount) {
  if (state.paused || termCount <= 1 || prefersReducedMotion) return;
  if (state.activeTerm >= termCount - 1) {
    state.paused = true;
    return;
  }
  termTimer = window.setTimeout(() => {
    state.activeTerm += 1;
    if (state.activeTerm >= termCount - 1) {
      state.paused = true;
    }
    renderVisualOnly();
  }, 900);
}

function handleKeydown(event) {
  if (
    event.target instanceof HTMLInputElement ||
    event.target instanceof HTMLTextAreaElement ||
    event.target instanceof HTMLButtonElement
  ) {
    return;
  }
  if (event.key === 'ArrowRight') {
    event.preventDefault();
    setStep(state.stepIndex + 1);
  } else if (event.key === 'ArrowLeft') {
    event.preventDefault();
    setStep(state.stepIndex - 1);
  } else if (event.key.toLowerCase() === 'r') {
    event.preventDefault();
    state.replayKey += 1;
    resetVisualState();
    renderVisualOnly();
  }
}

export function mountQwenViz(root) {
  app = root;
  state.stepIndex = 0;
  state.openGroups = new Set(['orientation']);
  state.replayKey = 0;
  state.mobileSidebarOpen = false;
  renderShell();
  const requestedStep = new URL(window.location.href).searchParams.get('step');
  const requestedIndex = QWEN_TUTORIAL.steps.findIndex(
    (step) => step.id === requestedStep,
  );
  setStep(requestedIndex >= 0 ? requestedIndex : 0);
  document.addEventListener('keydown', handleKeydown);

  return () => {
    window.clearTimeout(termTimer);
    termTimer = null;
    document.removeEventListener('keydown', handleKeydown);
    if (app === root) {
      app.innerHTML = '';
      app = null;
    }
  };
}
