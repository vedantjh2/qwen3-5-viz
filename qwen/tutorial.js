import {
  FFN_COLS,
  HIDDEN_COLS,
  MINI_CONFIG,
  MODEL_RUN,
  REAL_CONFIG,
  TOKENS,
  TOKEN_IDS,
  TOKEN_ROWS,
  VOCABULARY,
  elementwiseDetails,
  makeMatmulOperation,
  matrixData,
  rmsNormDetails,
} from './model.js';
import { formatNumber, transpose } from './math.js';

function rmsOperation(id, title, input, output, gamma) {
  return {
    id,
    tabLabel: 'RMSNorm',
    title,
    formula:
      'y[t,c] = x[t,c] / sqrt(mean_j(x[t,j]^2) + eps) x gamma[c]',
    caption:
      'RMSNorm rescales each token row independently. It does not mix tokens or channels.',
    input: matrixData(
      'residual stream',
      'X',
      input,
      'activation',
      TOKEN_ROWS,
      HIDDEN_COLS,
    ),
    output: matrixData(
      'normalized stream',
      'X_norm',
      output,
      'result',
      TOKEN_ROWS,
      HIDDEN_COLS,
    ),
    details: rmsNormDetails(input, gamma),
    defaultCell: [3, 2],
    auxiliaries: [
      matrixData(
        'learned scale',
        'gamma',
        [gamma],
        'weight',
        ['scale'],
        HIDDEN_COLS,
      ),
    ],
  };
}

function addFfnSteps(steps, groupId, groupTitle, layerIndex, ffn) {
  const prefix = `layer-${layerIndex}`;
  const normalizedData = matrixData(
    'normalized mixer residual',
    'U',
    ffn.normalized,
    'activation',
    TOKEN_ROWS,
    HIDDEN_COLS,
  );

  steps.push({
    id: `${prefix}-ffn-norm`,
    groupId,
    groupTitle,
    shortTitle: 'FFN RMSNorm',
    eyebrow: `Layer ${layerIndex} / dense FFN`,
    title: 'Normalize again before the SwiGLU feed-forward network',
    architectureTarget: `${prefix}-norm2`,
    paragraphs: [
      'Every Qwen3.5 decoder layer is pre-norm twice: once before the token mixer and once before the feed-forward network.',
      'This operation is row-local. The token positions do not communicate here.',
    ],
    formula: 'U = RMSNorm(X_after_mixer)',
    callouts: [
      ['Input shape', '4 x 8'],
      ['Token mixing', 'none'],
      ['Blue values', '8 learned scales'],
    ],
    visual: {
      kind: 'transform',
      operations: [
        rmsOperation(
          `${prefix}-ffn-rms`,
          'Post-mixer RMSNorm',
          ffn.input,
          ffn.normalized,
          ffn.normGamma,
        ),
      ],
    },
  });

  steps.push({
    id: `${prefix}-ffn-up`,
    groupId,
    groupTitle,
    shortTitle: 'Gate + up matmuls',
    eyebrow: `Layer ${layerIndex} / dense FFN`,
    title: 'Run the two parallel SwiGLU input projections',
    architectureTarget: `${prefix}-ffn`,
    paragraphs: [
      'The dense Qwen3.5 models do not route through experts. Every token uses the same two learned matrices.',
      'The two GEMMs can run in parallel, but Continue explains the gate projection first and the up projection second.',
    ],
    formula: 'G = U W_gate; P = U W_up',
    callouts: [
      ['Mini FFN width', '12'],
      ['Real 4B width', '9,216'],
      ['Parallel matmuls', '2'],
    ],
    visual: {
      kind: 'matmul',
      operations: [
        makeMatmulOperation({
          id: `${prefix}-gate-proj`,
          tabLabel: 'Gate projection',
          title: 'Gate branch',
          equation: 'G = U x W_gate',
          caption:
            'Each green gate activation is a dot product between one token row and one blue FFN weight column.',
          left: normalizedData,
          right: matrixData(
            'gate projection weights',
            'W_gate',
            ffn.gateWeight,
            'weight',
            HIDDEN_COLS,
            FFN_COLS,
          ),
          resultName: 'gate activations',
          resultSymbol: 'G',
          resultRowLabels: TOKEN_ROWS,
          resultColLabels: FFN_COLS,
          defaultCell: [3, 4],
        }),
        makeMatmulOperation({
          id: `${prefix}-up-proj`,
          tabLabel: 'Up projection',
          title: 'Value branch',
          equation: 'P = U x W_up',
          caption:
            'The second projection produces values that the activated gate will modulate.',
          left: normalizedData,
          right: matrixData(
            'up projection weights',
            'W_up',
            ffn.upWeight,
            'weight',
            HIDDEN_COLS,
            FFN_COLS,
          ),
          resultName: 'up activations',
          resultSymbol: 'P',
          resultRowLabels: TOKEN_ROWS,
          resultColLabels: FFN_COLS,
          defaultCell: [3, 4],
        }),
      ],
    },
  });

  steps.push({
    id: `${prefix}-ffn-swiglu`,
    groupId,
    groupTitle,
    shortTitle: 'SwiGLU',
    eyebrow: `Layer ${layerIndex} / dense FFN`,
    title: 'Apply SiLU to the gate and multiply element by element',
    architectureTarget: `${prefix}-ffn`,
    paragraphs: [
      'SwiGLU supplies the non-linearity. A gate can suppress or pass the matching up-branch value.',
      'There is no reduction here: every output uses values at the same row and column.',
    ],
    formula: 'H = SiLU(G) elementwise-multiply P',
    callouts: [
      ['Input shapes', '4 x 12'],
      ['Output shape', '4 x 12'],
      ['Operation', 'element-wise'],
    ],
    visual: {
      kind: 'elementwise',
      operator: 'swiglu',
      title: 'SwiGLU activation',
      formula: 'H[t,m] = SiLU(G[t,m]) x P[t,m]',
      caption:
        'Click a green result cell to see its gate and up-branch values.',
      left: matrixData(
        'gate activations',
        'G',
        ffn.gate,
        'gate',
        TOKEN_ROWS,
        FFN_COLS,
      ),
      right: matrixData(
        'up activations',
        'P',
        ffn.up,
        'activation',
        TOKEN_ROWS,
        FFN_COLS,
      ),
      result: matrixData(
        'SwiGLU hidden values',
        'H',
        ffn.activated,
        'result',
        TOKEN_ROWS,
        FFN_COLS,
      ),
      details: elementwiseDetails(
        ffn.gate,
        ffn.up,
        ffn.activated,
        'swiglu',
      ),
      defaultCell: [3, 4],
    },
  });

  steps.push({
    id: `${prefix}-ffn-down`,
    groupId,
    groupTitle,
    shortTitle: 'Down matmul',
    eyebrow: `Layer ${layerIndex} / dense FFN`,
    title: 'Project the wide SwiGLU representation back to residual width',
    architectureTarget: `${prefix}-ffn`,
    paragraphs: [
      'The down projection mixes all teaching-scale FFN channels into each residual-stream channel.',
      'In Qwen3.5-4B this is the much larger 9,216-to-2,560 matrix multiplication.',
    ],
    formula: 'D = H W_down',
    callouts: [
      ['Mini matmul', '(4 x 12) x (12 x 8)'],
      ['Output', '4 x 8'],
      ['Real inner size', '9,216'],
    ],
    visual: {
      kind: 'matmul',
      operations: [
        makeMatmulOperation({
          id: `${prefix}-down-proj`,
          tabLabel: 'Down projection',
          title: 'FFN down projection',
          equation: 'D = H x W_down',
          caption:
            'The selected result contains 12 products in the teaching model.',
          left: matrixData(
            'SwiGLU hidden values',
            'H',
            ffn.activated,
            'activation',
            TOKEN_ROWS,
            FFN_COLS,
          ),
          right: matrixData(
            'down projection weights',
            'W_down',
            ffn.downWeight,
            'weight',
            FFN_COLS,
            HIDDEN_COLS,
          ),
          resultName: 'FFN update',
          resultSymbol: 'D',
          resultRowLabels: TOKEN_ROWS,
          resultColLabels: HIDDEN_COLS,
          defaultCell: [3, 2],
        }),
      ],
    },
  });

  steps.push({
    id: `${prefix}-ffn-residual`,
    groupId,
    groupTitle,
    shortTitle: 'FFN residual',
    eyebrow: `Layer ${layerIndex} / residual stream`,
    title: 'Add the FFN update back into the residual stream',
    architectureTarget: `${prefix}-resid2`,
    paragraphs: [
      'The vertical residual path carries the existing representation. The FFN contributes an update rather than replacing it.',
      'After this coordinate-wise addition, the decoder layer is complete.',
    ],
    formula: 'X_out = X_after_mixer + D',
    callouts: [
      ['Left branch', 'residual stream'],
      ['Right branch', 'FFN update'],
      ['Output shape', '4 x 8'],
    ],
    visual: {
      kind: 'elementwise',
      operator: 'add',
      title: 'Second residual addition',
      formula: 'X_out[t,c] = X_after_mixer[t,c] + D[t,c]',
      caption: 'No dot product: matching green coordinates are added directly.',
      left: matrixData(
        'mixer residual',
        'X_after_mixer',
        ffn.input,
        'activation',
        TOKEN_ROWS,
        HIDDEN_COLS,
      ),
      right: matrixData(
        'FFN update',
        'D',
        ffn.down,
        'activation',
        TOKEN_ROWS,
        HIDDEN_COLS,
      ),
      result: matrixData(
        'layer output',
        'X_out',
        ffn.output,
        'result',
        TOKEN_ROWS,
        HIDDEN_COLS,
      ),
      details: elementwiseDetails(ffn.input, ffn.down, ffn.output, 'add'),
      defaultCell: [3, 2],
    },
  });
}

function buildDeltaSteps(run) {
  const layerIndex = run.layerIndex;
  const groupId = `delta-${layerIndex}`;
  const groupTitle = `Layer ${layerIndex}: Gated DeltaNet`;
  const prefix = `layer-${layerIndex}`;
  const steps = [];
  const valueHeads = 2;
  const valueHeadWidth = 2;
  const coreOutputCols = Array.from(
    { length: valueHeads * valueHeadWidth },
    (_, index) => `v${index}`,
  );
  const headOutputs = Array.from({ length: valueHeads }, (_, headIndex) => {
    const rows = TOKENS.map((_, tokenIndex) =>
      run.deltaTraces.find(
        (trace) =>
          trace.headIndex === headIndex && trace.tokenIndex === tokenIndex,
      ).read.values[0],
    );
    return matrixData(
      `value head ${headIndex} outputs across time`,
      `O_head${headIndex}`,
      rows,
      'result',
      TOKEN_ROWS,
      [`v${headIndex * 2}`, `v${headIndex * 2 + 1}`],
    );
  });
  const coreOutput = matrixData(
    'concatenated DeltaNet head outputs',
    'O_core',
    run.coreOutput,
    'result',
    TOKEN_ROWS,
    coreOutputCols,
  );
  const normalizedData = matrixData(
    'normalized stream',
    'N',
    run.normalized,
    'activation',
    TOKEN_ROWS,
    HIDDEN_COLS,
  );

  steps.push({
    id: `${prefix}-mixer-norm`,
    groupId,
    groupTitle,
    shortTitle: 'Mixer RMSNorm',
    eyebrow: `Layer ${layerIndex} / pre-norm`,
    title: 'Normalize each token before the Gated DeltaNet mixer',
    architectureTarget: `${prefix}-norm1`,
    paragraphs: [
      'Qwen3.5 is pre-norm: the residual stream is copied around the mixer while a normalized copy enters it.',
      'RMSNorm changes magnitudes but does not mix token positions.',
    ],
    formula: 'N = RMSNorm(X)',
    callouts: [
      ['Input', '4 tokens x 8 channels'],
      ['Rows independent', 'yes'],
      ['Residual copied', 'around mixer'],
    ],
    visual: {
      kind: 'transform',
      operations: [
        rmsOperation(
          `${prefix}-mixer-rms`,
          'Input RMSNorm',
          run.input,
          run.normalized,
          run.normGamma,
        ),
      ],
    },
  });

  steps.push({
    id: `${prefix}-qkv`,
    groupId,
    groupTitle,
    shortTitle: 'Packed QKV matmul',
    eyebrow: `Layer ${layerIndex} / Gated DeltaNet`,
    title: 'Project every token into packed query, key, and value channels',
    architectureTarget: `${prefix}-mixer`,
    paragraphs: [
      'The implementation uses one packed matrix multiplication for Q, K, and V, then splits the output columns.',
      'The mini model preserves the real 1:2 Q/K-to-V head ratio.',
    ],
    formula: '[Q_raw | K_raw | V_raw] = N W_qkv',
    callouts: [
      ['Mini matmul', '(4 x 8) x (8 x 8)'],
      ['Packed columns', '2 Q + 2 K + 4 V'],
      ['Real packed width', '8,192'],
    ],
    visual: {
      kind: 'matmul',
      operations: [
        makeMatmulOperation({
          id: `${prefix}-qkv-proj`,
          tabLabel: 'Packed QKV',
          title: 'Packed DeltaNet projection',
          equation: 'QKV_raw = N x W_qkv',
          caption:
            'Click a green Q, K, or V cell. The animation pairs its input row with the matching blue weight column.',
          left: normalizedData,
          right: matrixData(
            'packed QKV weights',
            'W_qkv',
            run.qkvWeight,
            'weight',
            HIDDEN_COLS,
            ['q0', 'q1', 'k0', 'k1', 'v0', 'v1', 'v2', 'v3'],
          ),
          resultName: 'packed QKV projection',
          resultSymbol: 'QKV_raw',
          resultRowLabels: TOKEN_ROWS,
          resultColLabels: [
            'q0',
            'q1',
            'k0',
            'k1',
            'v0',
            'v1',
            'v2',
            'v3',
          ],
          defaultCell: [3, 1],
        }),
      ],
    },
  });

  steps.push({
    id: `${prefix}-controls`,
    groupId,
    groupTitle,
    shortTitle: 'z, beta, g matmuls',
    eyebrow: `Layer ${layerIndex} / Gated DeltaNet`,
    title: 'Compute output gate, write rate, and state decay controls',
    architectureTarget: `${prefix}-mixer`,
    paragraphs: [
      'Three additional learned projections control what DeltaNet remembers and emits.',
      'They can be evaluated from the same normalized input; Continue explains z, beta, and g one at a time.',
    ],
    formula:
      'z = N W_z; beta = sigmoid(N W_b); g = -exp(A_log) softplus(N W_a + dt_bias)',
    callouts: [
      ['z width', '4 value channels'],
      ['beta range', '0 to 1'],
      ['exp(g) range', '0 to 1'],
    ],
    visual: {
      kind: 'matmul',
      operations: [
        makeMatmulOperation({
          id: `${prefix}-z-proj`,
          tabLabel: 'Output gate z',
          title: 'Output-gate projection',
          equation: 'z = N x W_z',
          caption:
            'z stays at value-head width and later passes through SiLU.',
          left: normalizedData,
          right: matrixData(
            'z weights',
            'W_z',
            run.zWeight,
            'weight',
            HIDDEN_COLS,
            ['z0', 'z1', 'z2', 'z3'],
          ),
          resultName: 'output gate logits',
          resultSymbol: 'z',
          resultRowLabels: TOKEN_ROWS,
          resultColLabels: ['z0', 'z1', 'z2', 'z3'],
          defaultCell: [3, 1],
        }),
        makeMatmulOperation({
          id: `${prefix}-beta-proj`,
          tabLabel: 'Write rate beta',
          title: 'Write-rate projection',
          equation: 'b = N x W_b; beta = sigmoid(b)',
          caption:
            'The raw green dot product b becomes a bounded write rate.',
          left: normalizedData,
          right: matrixData(
            'beta weights',
            'W_b',
            run.betaWeight,
            'weight',
            HIDDEN_COLS,
            ['head0', 'head1'],
          ),
          resultName: 'write-rate logits',
          resultSymbol: 'b',
          resultRowLabels: TOKEN_ROWS,
          resultColLabels: ['head0', 'head1'],
          defaultCell: [3, 0],
          derived: {
            matrix: matrixData(
              'update rates',
              'beta',
              run.beta,
              'gate',
              TOKEN_ROWS,
              ['head0', 'head1'],
            ),
            formula: 'beta = sigmoid(b)',
            caption:
              'Each computed beta says how aggressively this token corrects a value head.',
          },
        }),
        makeMatmulOperation({
          id: `${prefix}-decay-proj`,
          tabLabel: 'Decay g',
          title: 'Decay projection',
          equation: 'a = N x W_a',
          caption:
            'a is combined with learned per-head A_log and dt_bias values.',
          left: normalizedData,
          right: matrixData(
            'decay weights',
            'W_a',
            run.decayWeight,
            'weight',
            HIDDEN_COLS,
            ['head0', 'head1'],
          ),
          resultName: 'decay logits',
          resultSymbol: 'a',
          resultRowLabels: TOKEN_ROWS,
          resultColLabels: ['head0', 'head1'],
          defaultCell: [3, 0],
          derived: {
            matrix: matrixData(
              'log decays',
              'g',
              run.decay,
              'gate',
              TOKEN_ROWS,
              ['head0', 'head1'],
            ),
            formula: 'g = -exp(A_log) x softplus(a + dt_bias)',
            caption:
              'g is negative, so exp(g) is a state-retention factor from zero to one.',
          },
        }),
      ],
    },
  });

  steps.push({
    id: `${prefix}-conv`,
    groupId,
    groupTitle,
    shortTitle: 'Causal depthwise conv',
    eyebrow: `Layer ${layerIndex} / local sequence mixing`,
    title: 'Run a four-tap causal depthwise convolution over packed QKV',
    architectureTarget: `${prefix}-mixer`,
    paragraphs: [
      'Each projected Q/K/V channel receives a tiny causal convolution and SiLU activation before the recurrent scan.',
      'Depthwise means channels stay separate. Causal means only current and earlier token positions are read.',
    ],
    formula:
      'QKV_conv[t,c] = SiLU(sum_r QKV_raw[t-r,c] kernel[c,r])',
    callouts: [
      ['Kernel length', '4 (same as real 4B)'],
      ['Channel mixing', 'none'],
      ['Future tokens', 'never read'],
    ],
    visual: {
      kind: 'convolution',
      input: matrixData(
        'projected QKV',
        'QKV_raw',
        run.qkvProjected,
        'activation',
        TOKEN_ROWS,
        ['q0', 'q1', 'k0', 'k1', 'v0', 'v1', 'v2', 'v3'],
      ),
      kernels: matrixData(
        'depthwise kernels',
        'K_conv',
        run.convKernels,
        'weight',
        ['q0', 'q1', 'k0', 'k1', 'v0', 'v1', 'v2', 'v3'],
        ['oldest', '-2', '-1', 'current'],
      ),
      output: matrixData(
        'convolved QKV',
        'QKV_conv',
        run.convolvedQkv,
        'result',
        TOKEN_ROWS,
        ['q0', 'q1', 'k0', 'k1', 'v0', 'v1', 'v2', 'v3'],
      ),
      traces: run.convTraces,
      defaultCell: [3, 1],
    },
  });

  steps.push({
    id: `${prefix}-scan`,
    groupId,
    groupTitle,
    shortTitle: 'Delta state scan',
    eyebrow: `Layer ${layerIndex} / recurrent memory`,
    title: 'Predict, correct, write, and read the fixed-size DeltaNet state',
    architectureTarget: `${prefix}-mixer`,
    paragraphs: [
      'Each value head carries a fixed key-by-value state through time instead of building a token-by-token attention matrix.',
      'The key reads the old prediction, beta scales the error, a rank-one outer product writes the correction, and the query reads the updated state. Production prefill uses an equivalent chunked implementation.',
    ],
    formula:
      'S <- exp(g)S; delta <- beta(v - kS); S <- S + k^T delta; o <- qS',
    callouts: [
      ['Mini state / head', '2 x 2'],
      ['Real state / V head', '128 x 128'],
      ['State grows with T?', 'no'],
    ],
    visual: {
      kind: 'delta',
      tokens: TOKENS,
      traces: run.deltaTraces,
      heads: valueHeads,
      headOutputs,
      coreOutput,
    },
  });

  const gatedDetails = run.gatedOutput.map((row, rowIndex) =>
    row.map((value, colIndex) => {
      const headIndex = Math.floor(colIndex / 2);
      const headValues = run.coreOutput[rowIndex].slice(
        headIndex * 2,
        headIndex * 2 + 2,
      );
      const meanSquare =
        headValues.reduce((sum, item) => sum + item * item, 0) / 2;
      const zValue = run.z[rowIndex][colIndex];
      return `RMSNorm(head) x SiLU(z=${formatNumber(zValue)}) = ${formatNumber(value)}; head mean-square=${formatNumber(meanSquare)}`;
    }),
  );

  steps.push({
    id: `${prefix}-gated-norm`,
    groupId,
    groupTitle,
    shortTitle: 'Gated head norm',
    eyebrow: `Layer ${layerIndex} / Gated DeltaNet`,
    title: 'Normalize each value head and apply the learned z gate',
    architectureTarget: `${prefix}-mixer`,
    paragraphs: [
      'The previous step stacked every q_t S_t read into O_head0 and O_head1, then concatenated them into O_core. No values appear from nowhere here.',
      'Split each O_core row back into its two value-head slices, RMS-normalize each slice, then multiply by the matching SiLU(z) slice.',
      'The z projection decides how much of each computed state-read channel leaves the recurrent mixer.',
    ],
    formula: 'O_gated = RMSNorm(O_core) elementwise-multiply SiLU(z)',
    callouts: [
      ['Head width', '2 mini / 128 real'],
      ['Value heads', '2 mini / 32 real'],
      ['Output width', '4 mini / 4,096 real'],
    ],
    visual: {
      kind: 'transform',
      operations: [
        {
          id: `${prefix}-gated-rms`,
          tabLabel: 'Gate heads',
          title: 'Per-head RMSNorm and z gate',
          formula:
            'O_gated[t,h,d] = RMSNorm(O_core[t,h,:])[d] x SiLU(z[t,h,d])',
          caption:
            'The four columns are two adjacent, two-wide value heads.',
          input: matrixData(
            'DeltaNet core output',
            'O_core',
            run.coreOutput,
            'activation',
            TOKEN_ROWS,
            coreOutputCols,
          ),
          output: matrixData(
            'gated head output',
            'O_gated',
            run.gatedOutput,
            'result',
            TOKEN_ROWS,
            coreOutputCols,
          ),
          details: gatedDetails,
          defaultCell: [3, 1],
          auxiliaries: [
            matrixData(
              'z gate',
              'z',
              run.z,
              'gate',
              TOKEN_ROWS,
              coreOutputCols,
            ),
            matrixData(
              'head scale',
              'gamma_head',
              [run.gatedGamma],
              'weight',
              ['scale'],
              ['v0', 'v1'],
            ),
          ],
        },
      ],
    },
  });

  steps.push({
    id: `${prefix}-out-proj`,
    groupId,
    groupTitle,
    shortTitle: 'Mixer output matmul',
    eyebrow: `Layer ${layerIndex} / Gated DeltaNet`,
    title: 'Project concatenated value heads back to residual width',
    architectureTarget: `${prefix}-mixer`,
    paragraphs: [
      'The recurrent mixer is still at value-head width. One learned output matrix mixes those channels back into the residual stream.',
      'This is the DeltaNet counterpart of the O projection after full attention.',
    ],
    formula: 'M = O_gated W_out',
    callouts: [
      ['Mini matmul', '(4 x 4) x (4 x 8)'],
      ['Real input width', '4,096'],
      ['Output width', '2,560 real'],
    ],
    visual: {
      kind: 'matmul',
      operations: [
        makeMatmulOperation({
          id: `${prefix}-delta-out`,
          tabLabel: 'Output projection',
          title: 'DeltaNet output projection',
          equation: 'M = O_gated x W_out',
          caption:
            'Each green residual update cell mixes all value-head channels through blue weights.',
          left: matrixData(
            'gated head output',
            'O_gated',
            run.gatedOutput,
            'activation',
            TOKEN_ROWS,
            coreOutputCols,
          ),
          right: matrixData(
            'DeltaNet output weights',
            'W_out',
            run.outWeight,
            'weight',
            coreOutputCols,
            HIDDEN_COLS,
          ),
          resultName: 'mixer update',
          resultSymbol: 'M',
          resultRowLabels: TOKEN_ROWS,
          resultColLabels: HIDDEN_COLS,
          defaultCell: [3, 2],
        }),
      ],
    },
  });

  steps.push({
    id: `${prefix}-mixer-residual`,
    groupId,
    groupTitle,
    shortTitle: 'Mixer residual',
    eyebrow: `Layer ${layerIndex} / residual stream`,
    title: 'Add the DeltaNet update to the untouched residual branch',
    architectureTarget: `${prefix}-resid1`,
    paragraphs: [
      'The normalized path produced an update M. The original layer input bypassed all that work and is added back cell by cell.',
      'The green residual highway carries information directly through all decoder layers.',
    ],
    formula: 'X_after_mixer = X_in + M',
    callouts: [
      ['Shape', '4 x 8'],
      ['Operation', 'element-wise add'],
      ['Blue weights', 'none in this step'],
    ],
    visual: {
      kind: 'elementwise',
      operator: 'add',
      title: 'First residual addition',
      formula: 'X_after_mixer[t,c] = X_in[t,c] + M[t,c]',
      caption:
        'Select a green output coordinate to see the two computed values that meet.',
      left: matrixData(
        'layer input',
        'X_in',
        run.input,
        'activation',
        TOKEN_ROWS,
        HIDDEN_COLS,
      ),
      right: matrixData(
        'mixer update',
        'M',
        run.mixerOutput,
        'activation',
        TOKEN_ROWS,
        HIDDEN_COLS,
      ),
      result: matrixData(
        'mixer residual',
        'X_after_mixer',
        run.residualOutput,
        'result',
        TOKEN_ROWS,
        HIDDEN_COLS,
      ),
      details: elementwiseDetails(
        run.input,
        run.mixerOutput,
        run.residualOutput,
        'add',
      ),
      defaultCell: [3, 2],
    },
  });

  addFfnSteps(steps, groupId, groupTitle, layerIndex, run.ffn);
  return steps;
}

function buildDeltaRepeatStep(layer0, layer1, layer2) {
  return {
    id: 'repeat-delta-layers',
    groupId: 'delta-repeat',
    groupTitle: 'Layers 1-2: repeat Gated DeltaNet',
    shortTitle: 'Fast-forward two GDN layers',
    eyebrow: 'Layers 1 and 2 / repeated architecture',
    title: 'Run the same Gated DeltaNet plus SwiGLU graph two more times',
    architectureTarget: 'repeat-delta',
    paragraphs: [
      'Layers 1 and 2 use exactly the operation sequence explained for layer 0, so repeating every projection and state-cell animation would add length without adding a new concept.',
      'They do not reuse layer 0 weights or its recurrent state. Each layer has its own learned matrices and its own pair of value-head state chains, while the green residual output from one layer becomes the input to the next.',
    ],
    formula:
      'X_2 = GDNLayer_2(GDNLayer_1(X_0_out)); each layer includes its own dense SwiGLU FFN',
    sequenceNote:
      'This is a fast-forward through two real decoder layers, not a shortcut in the model. The arithmetic still runs in the teaching model; only the repeated explanation is collapsed.',
    callouts: [
      ['Repeated layers', '1 and 2'],
      ['Weights shared?', 'no'],
      ['State shared across layers?', 'no'],
    ],
    visual: {
      kind: 'delta-repeat',
      stages: [
        matrixData(
          'detailed layer 0 output',
          'X_after_L0',
          layer0.ffn.output,
          'result',
          TOKEN_ROWS,
          HIDDEN_COLS,
        ),
        matrixData(
          'layer 1 output',
          'X_after_L1',
          layer1.ffn.output,
          'result',
          TOKEN_ROWS,
          HIDDEN_COLS,
        ),
        matrixData(
          'layer 2 output',
          'X_after_L2',
          layer2.ffn.output,
          'result',
          TOKEN_ROWS,
          HIDDEN_COLS,
        ),
      ],
      transitions: [
        {
          layer: 1,
          label: 'same GDN + FFN graph',
          detail: 'new W^(1), new S^(1)',
        },
        {
          layer: 2,
          label: 'same GDN + FFN graph',
          detail: 'new W^(2), new S^(2)',
        },
      ],
    },
  };
}

function buildAttentionSteps(run) {
  const layerIndex = run.layerIndex;
  const groupId = 'full-attention';
  const groupTitle = `Layer ${layerIndex}: Gated full attention`;
  const prefix = `layer-${layerIndex}`;
  const steps = [];
  const queryHeads = MINI_CONFIG.attentionQueryHeads;
  const headDim = MINI_CONFIG.attentionHeadDim;
  const headCols = Array.from({ length: headDim }, (_, index) => `d${index}`);
  const queryCols = Array.from(
    { length: queryHeads },
    (_, headIndex) =>
      Array.from(
        { length: headDim },
        (_, dimIndex) => `q${headIndex}.${dimIndex}`,
      ),
  ).flat();
  const gateCols = Array.from(
    { length: queryHeads },
    (_, headIndex) =>
      Array.from(
        { length: headDim },
        (_, dimIndex) => `g${headIndex}.${dimIndex}`,
      ),
  ).flat();
  const qGateCols = [...queryCols, ...gateCols];
  const keyCols = headCols.map((_, index) => `k${index}`);
  const valueCols = headCols.map((_, index) => `v${index}`);
  const attentionScale = 1 / Math.sqrt(headDim);
  const normalizedData = matrixData(
    'normalized stream',
    'N',
    run.normalized,
    'activation',
    TOKEN_ROWS,
    HIDDEN_COLS,
  );

  steps.push({
    id: `${prefix}-mixer-norm`,
    groupId,
    groupTitle,
    shortTitle: 'Mixer RMSNorm',
    eyebrow: `Layer ${layerIndex} / pre-norm`,
    title: 'Normalize before the fourth layer switches to full attention',
    architectureTarget: `${prefix}-norm1`,
    paragraphs: [
      'The decoder-layer shell is unchanged. Only the token mixer swaps from Gated DeltaNet to gated grouped-query attention.',
      'The residual branch still bypasses the normalized mixer path.',
    ],
    formula: 'N = RMSNorm(X)',
    callouts: [
      ['Layer type', 'full attention'],
      ['Pattern position', '4th of 4'],
      ['Residual copied', 'around mixer'],
    ],
    visual: {
      kind: 'transform',
      operations: [
        rmsOperation(
          `${prefix}-mixer-rms`,
          'Attention input RMSNorm',
          run.input,
          run.normalized,
          run.normGamma,
        ),
      ],
    },
  });

  steps.push({
    id: `${prefix}-qkv`,
    groupId,
    groupTitle,
    shortTitle: 'Q/gate, K, V matmuls',
    eyebrow: `Layer ${layerIndex} / grouped-query attention`,
    title: 'Project queries plus output gates, keys, and values',
    architectureTarget: `${prefix}-mixer`,
    paragraphs: [
      'The Q projection is packed with an equally wide output gate. K and V use separate, narrower matrices.',
      'The teaching model preserves Qwen3.5-4B\'s 4:1 grouped-query ratio with four Q heads and one K/V head.',
    ],
    formula: '[Q | gate] = N W_qg; K = N W_k; V = N W_v',
    callouts: [
      ['Q heads', '4 mini / 16 real'],
      ['K/V heads', '1 mini / 4 real'],
      ['Head width', '2 mini / 256 real'],
    ],
    visual: {
      kind: 'matmul',
      operations: [
        makeMatmulOperation({
          id: `${prefix}-q-gate`,
          tabLabel: 'Q + output gate',
          title: 'Packed query and output-gate projection',
          equation: '[Q | gate] = N x W_qg',
          caption:
            'The first eight green columns become four 2-wide Q heads; the next eight become matching gate logits.',
          left: normalizedData,
          right: matrixData(
            'packed query/gate weights',
            'W_qg',
            run.qGateWeight,
            'weight',
            HIDDEN_COLS,
            qGateCols,
          ),
          resultName: 'packed Q and gate',
          resultSymbol: 'QG',
          resultRowLabels: TOKEN_ROWS,
          resultColLabels: qGateCols,
          defaultCell: [3, 1],
        }),
        makeMatmulOperation({
          id: `${prefix}-k`,
          tabLabel: 'K projection',
          title: 'Shared key projection',
          equation: 'K = N x W_k',
          caption: 'One mini key head is shared by all four query heads.',
          left: normalizedData,
          right: matrixData(
            'key weights',
            'W_k',
            run.kWeight,
            'weight',
            HIDDEN_COLS,
            keyCols,
          ),
          resultName: 'keys',
          resultSymbol: 'K',
          resultRowLabels: TOKEN_ROWS,
          resultColLabels: keyCols,
          defaultCell: [3, 1],
        }),
        makeMatmulOperation({
          id: `${prefix}-v`,
          tabLabel: 'V projection',
          title: 'Shared value projection',
          equation: 'V = N x W_v',
          caption: 'The same grouped sharing pattern applies to values.',
          left: normalizedData,
          right: matrixData(
            'value weights',
            'W_v',
            run.vWeight,
            'weight',
            HIDDEN_COLS,
            valueCols,
          ),
          resultName: 'values',
          resultSymbol: 'V',
          resultRowLabels: TOKEN_ROWS,
          resultColLabels: valueCols,
          defaultCell: [3, 1],
        }),
      ],
    },
  });

  const ropeOperations = [];
  for (let headIndex = 0; headIndex < queryHeads; headIndex += 1) {
    ropeOperations.push({
      id: `${prefix}-q-rope-${headIndex}`,
      tabLabel: `Q head ${headIndex}`,
      title: `Normalize and rotate query head ${headIndex}`,
      formula: 'Q_hat = partial_RoPE(RMSNorm(Q))',
      caption:
        'The mini head is one visible 2D rotary pair. The real model rotates 64 of each 256-dimensional head.',
      input: matrixData(
        'query before RoPE',
        `Q${headIndex}`,
        run.qBeforeRope[headIndex],
        'activation',
        TOKEN_ROWS,
        headCols,
      ),
      output: matrixData(
        'query after RoPE',
        `Q${headIndex}_rope`,
        run.qAfterRope[headIndex],
        'result',
        TOKEN_ROWS,
        headCols,
      ),
      details: run.qAfterRope[headIndex].map((row, tokenIndex) =>
        row.map((value, colIndex) =>
          colIndex < MINI_CONFIG.rotaryDimensions
            ? `Token ${tokenIndex}: rotate dimensions 0-1; output=${formatNumber(value)}`
            : `Dimension ${colIndex} is outside the rotary slice; output=${formatNumber(value)}`,
        ),
      ),
      defaultCell: [3, 1],
      auxiliaries: [
        matrixData(
          'Q head RMS scale',
          'gamma_q',
          [run.qNormGamma],
          'weight',
          ['scale'],
          headCols,
        ),
      ],
    });
  }
  ropeOperations.push({
    id: `${prefix}-k-rope`,
    tabLabel: 'Shared K',
    title: 'Normalize and rotate the shared key head',
    formula: 'K_hat = partial_RoPE(RMSNorm(K))',
    caption: 'This one rotated K head is reused by all four mini query heads.',
    input: matrixData(
      'key before RoPE',
      'K',
      run.kBeforeRope[0],
      'activation',
      TOKEN_ROWS,
      headCols,
    ),
    output: matrixData(
      'key after RoPE',
      'K_rope',
      run.kAfterRope[0],
      'result',
      TOKEN_ROWS,
      headCols,
    ),
    details: run.kAfterRope[0].map((row, tokenIndex) =>
      row.map((value, colIndex) =>
        colIndex < MINI_CONFIG.rotaryDimensions
          ? `Token ${tokenIndex}: rotate shared K dimensions 0-1; output=${formatNumber(value)}`
          : `Dimension ${colIndex} is unchanged; output=${formatNumber(value)}`,
      ),
    ),
    defaultCell: [3, 1],
    auxiliaries: [
      matrixData(
        'K head RMS scale',
        'gamma_k',
        [run.kNormGamma],
        'weight',
        ['scale'],
        headCols,
      ),
    ],
  });

  steps.push({
    id: `${prefix}-rope`,
    groupId,
    groupTitle,
    shortTitle: 'Q/K norm + partial RoPE',
    eyebrow: `Layer ${layerIndex} / position information`,
    title: 'Normalize Q and K heads, then rotate only part of each head',
    architectureTarget: `${prefix}-mixer`,
    paragraphs: [
      'Full-attention layers carry position through rotary embeddings. Qwen3.5-4B rotates 64 of each 256-dimensional head.',
      'DeltaNet layers do not use RoPE; their causal convolution and recurrent scan already encode order.',
    ],
    formula:
      'Q_hat, K_hat = partial_RoPE(head_RMSNorm(Q), head_RMSNorm(K))',
    callouts: [
      ['Mini rotary slice', '2 of 2 dimensions'],
      ['Real rotary slice', '64 of 256'],
      ['Applied to V?', 'no'],
    ],
    visual: { kind: 'transform', operations: ropeOperations },
  });

  steps.push({
    id: `${prefix}-gqa`,
    groupId,
    groupTitle,
    shortTitle: 'Grouped-query sharing',
    eyebrow: `Layer ${layerIndex} / grouped-query attention`,
    title: 'Reuse one K/V head for all four query heads',
    architectureTarget: `${prefix}-mixer`,
    paragraphs: [
      'Grouped-query attention reduces K/V cache memory. Query heads stay distinct while groups share K and V.',
      'The mini ratio is 4:1, exactly matching Qwen3.5-4B\'s 16 Q heads and 4 K/V heads.',
    ],
    formula: 'K_h = K_group(h); V_h = V_group(h)',
    callouts: [
      ['Mini sharing ratio', '4 Q per K/V'],
      ['Real sharing ratio', '4 Q per K/V'],
      ['Cache benefit', 'fewer K/V heads'],
    ],
    visual: {
      kind: 'gqa',
      qHeads: queryHeads,
      kvHeads: 1,
      q: run.qAfterRope.map((head, index) =>
        matrixData(
          `query head ${index}`,
          `Q${index}`,
          head,
          'activation',
          TOKEN_ROWS,
          headCols,
        ),
      ),
      k: [
        matrixData(
          'shared key head',
          'K0',
          run.kAfterRope[0],
          'activation',
          TOKEN_ROWS,
          headCols,
        ),
      ],
      v: [
        matrixData(
          'shared value head',
          'V0',
          run.vHeads[0],
          'activation',
          TOKEN_ROWS,
          headCols,
        ),
      ],
    },
  });

  steps.push({
    id: `${prefix}-scores`,
    groupId,
    groupTitle,
    shortTitle: 'Q K-transpose matmul',
    eyebrow: `Layer ${layerIndex} / attention scores`,
    title: 'Compare every query with every key',
    architectureTarget: `${prefix}-mixer`,
    paragraphs: [
      'Each score is a dot product between one query-token row and one key-token row.',
      'This T-by-T matrix is the quadratic structure that the three DeltaNet layers avoided.',
    ],
    formula: 'Scores_h = Q_h K_h^T / sqrt(head_dim)',
    callouts: [
      ['Mini score matrix', '4 x 4 per head'],
      ['Real sequence scaling', 'T x T'],
      ['K/V sharing', 'same K, different Q'],
    ],
    visual: {
      kind: 'matmul',
      operations: Array.from({ length: queryHeads }, (_, headIndex) =>
        makeMatmulOperation({
          id: `${prefix}-scores-head${headIndex}`,
          tabLabel: `Head ${headIndex} scores`,
          title: `Scaled query-key scores, head ${headIndex}`,
          equation: `Scores_${headIndex} = Q_${headIndex} x K_0^T / sqrt(${headDim})`,
          caption:
            `The trace includes the 1/sqrt(${headDim}) scale on every dot-product term.`,
          left: matrixData(
            `query head ${headIndex}`,
            `Q${headIndex}`,
            run.qAfterRope[headIndex],
            'activation',
            TOKEN_ROWS,
            headCols,
          ),
          right: matrixData(
            'transposed shared key',
            'K0^T',
            transpose(run.kAfterRope[headIndex]),
            'activation',
            headCols,
            TOKEN_ROWS,
          ),
          resultName: 'scaled scores',
          resultSymbol: `Scores${headIndex}`,
          resultRowLabels: TOKEN_ROWS,
          resultColLabels: TOKEN_ROWS,
          termScale: attentionScale,
          defaultCell: [3, 1],
        }),
      ),
    },
  });

  for (let headIndex = 0; headIndex < queryHeads; headIndex += 1) {
    steps.push({
      id: `${prefix}-softmax-head${headIndex}`,
      groupId,
      groupTitle,
      shortTitle: `Head ${headIndex} mask + softmax`,
      eyebrow: `Layer ${layerIndex} / attention probabilities`,
      title: `Mask and normalize attention scores for query head ${headIndex}`,
      architectureTarget: `${prefix}-mixer`,
      paragraphs: [
        `Query head ${headIndex} has its own T x T score matrix even though all four query heads share the same K/V head.`,
        'The causal mask removes future positions, then softmax turns each remaining score row into probabilities that sum to one.',
      ],
      formula: `P_${headIndex}[t,:] = softmax(mask_causal(Scores_${headIndex}[t,:]))`,
      callouts: [
        ['Query head', `${headIndex} of ${queryHeads - 1}`],
        ['Future probability', '0'],
        ['Row sum after softmax', '1.000'],
      ],
      visual: {
        kind: 'softmax',
        headIndex,
        totalHeads: queryHeads,
        scores: matrixData(
          'raw scaled scores',
          `Scores${headIndex}`,
          run.scores[headIndex],
          'activation',
          TOKEN_ROWS,
          TOKEN_ROWS,
        ),
        maskedScores: matrixData(
          'causally masked scores',
          `Masked${headIndex}`,
          run.maskedScores[headIndex],
          'mask',
          TOKEN_ROWS,
          TOKEN_ROWS,
        ),
        probabilities: matrixData(
          'attention probabilities',
          `P${headIndex}`,
          run.probabilities[headIndex],
          'probability',
          TOKEN_ROWS,
          TOKEN_ROWS,
        ),
        defaultRow: 3,
      },
    });
  }

  steps.push({
    id: `${prefix}-values`,
    groupId,
    groupTitle,
    shortTitle: 'P V context matmul',
    eyebrow: `Layer ${layerIndex} / attention values`,
    title: 'Mix value rows with the attention probabilities',
    architectureTarget: `${prefix}-mixer`,
    paragraphs: [
      'Each context cell is a dot product between one probability row and one value-feature column.',
      'Tokens with larger probabilities contribute more of their V vectors.',
    ],
    formula: 'Context_h = P_h V_h',
    callouts: [
      ['Mini matmul', '(4 x 4) x (4 x 2)'],
      ['Future rows used', 'none'],
      ['Heads', 'independent'],
    ],
    visual: {
      kind: 'matmul',
      operations: Array.from({ length: queryHeads }, (_, headIndex) =>
        makeMatmulOperation({
          id: `${prefix}-context-head${headIndex}`,
          tabLabel: `Head ${headIndex} context`,
          title: `Probability-weighted values, head ${headIndex}`,
          equation: `Context_${headIndex} = P_${headIndex} x V_0`,
          caption:
            'The green probability row supplies coefficients for the selected value feature.',
          left: matrixData(
            `attention probabilities, head ${headIndex}`,
            `P${headIndex}`,
            run.probabilities[headIndex],
            'probability',
            TOKEN_ROWS,
            TOKEN_ROWS,
          ),
          right: matrixData(
            'shared values',
            'V0',
            run.vHeads[headIndex],
            'activation',
            TOKEN_ROWS,
            headCols,
          ),
          resultName: `head ${headIndex} context`,
          resultSymbol: `C${headIndex}`,
          resultRowLabels: TOKEN_ROWS,
          resultColLabels: headCols,
          defaultCell: [3, 1],
        }),
      ),
    },
  });

  steps.push({
    id: `${prefix}-attn-gate`,
    groupId,
    groupTitle,
    shortTitle: 'Attention output gate',
    eyebrow: `Layer ${layerIndex} / gated attention`,
    title: 'Concatenate the heads and apply the packed sigmoid gate',
    architectureTarget: `${prefix}-mixer`,
    paragraphs: [
      'Qwen3.5 adds an output gate to grouped-query attention. Its logits came from the packed Q projection.',
      'Each context channel is multiplied by its own sigmoid gate before the output projection.',
    ],
    formula:
      'C_gated = concat(C_0, C_1, C_2, C_3) elementwise-multiply sigmoid(gate)',
    callouts: [
      ['Context width', '8'],
      ['Gate range', '0 to 1'],
      ['Operation', 'element-wise'],
    ],
    visual: {
      kind: 'elementwise',
      operator: 'gate',
      title: 'Gated attention context',
      formula: 'C_gated[t,c] = C[t,c] x sigmoid(gate[t,c])',
      caption:
        'Click a green output to inspect the context value and gate logit.',
      left: matrixData(
        'concatenated context',
        'C',
        run.context,
        'activation',
        TOKEN_ROWS,
        HIDDEN_COLS,
      ),
      right: matrixData(
        'output gate logits',
        'gate',
        run.outputGate,
        'gate',
        TOKEN_ROWS,
        HIDDEN_COLS,
      ),
      result: matrixData(
        'gated context',
        'C_gated',
        run.gatedContext,
        'result',
        TOKEN_ROWS,
        HIDDEN_COLS,
      ),
      details: elementwiseDetails(
        run.context,
        run.outputGate,
        run.gatedContext,
        'gate',
      ),
      defaultCell: [3, 2],
    },
  });

  steps.push({
    id: `${prefix}-out-proj`,
    groupId,
    groupTitle,
    shortTitle: 'Attention O matmul',
    eyebrow: `Layer ${layerIndex} / gated attention`,
    title: 'Mix the gated heads through the output projection',
    architectureTarget: `${prefix}-mixer`,
    paragraphs: [
      'The O projection recombines channels from all query heads into one residual-width update.',
      'This is the final matrix multiplication inside the attention mixer.',
    ],
    formula: 'M = C_gated W_o',
    callouts: [
      ['Mini matmul', '(4 x 8) x (8 x 8)'],
      ['Real O weight', '(4096 x 2560)'],
      ['Output', '4 x 8'],
    ],
    visual: {
      kind: 'matmul',
      operations: [
        makeMatmulOperation({
          id: `${prefix}-attention-out`,
          tabLabel: 'O projection',
          title: 'Attention output projection',
          equation: 'M = C_gated x W_o',
          caption:
            'Every green result cell is a dot product over all concatenated head channels.',
          left: matrixData(
            'gated context',
            'C_gated',
            run.gatedContext,
            'activation',
            TOKEN_ROWS,
            HIDDEN_COLS,
          ),
          right: matrixData(
            'attention output weights',
            'W_o',
            run.outWeight,
            'weight',
            HIDDEN_COLS,
            HIDDEN_COLS,
          ),
          resultName: 'attention update',
          resultSymbol: 'M',
          resultRowLabels: TOKEN_ROWS,
          resultColLabels: HIDDEN_COLS,
          defaultCell: [3, 2],
        }),
      ],
    },
  });

  steps.push({
    id: `${prefix}-mixer-residual`,
    groupId,
    groupTitle,
    shortTitle: 'Mixer residual',
    eyebrow: `Layer ${layerIndex} / residual stream`,
    title: 'Add the full-attention update to the residual stream',
    architectureTarget: `${prefix}-resid1`,
    paragraphs: [
      'The internal mixer was different, but the layer rejoins the same green residual backbone.',
      'The representation now contains three recurrent mixing passes followed by one explicit all-pairs attention pass.',
    ],
    formula: 'X_after_mixer = X_in + M',
    callouts: [
      ['Mixer', 'gated full attention'],
      ['Operation', 'element-wise add'],
      ['Quartet status', 'token mixing complete'],
    ],
    visual: {
      kind: 'elementwise',
      operator: 'add',
      title: 'Attention residual addition',
      formula: 'X_after_mixer[t,c] = X_in[t,c] + M[t,c]',
      caption:
        'The residual branch and attention update meet at matching coordinates.',
      left: matrixData(
        'layer input',
        'X_in',
        run.input,
        'activation',
        TOKEN_ROWS,
        HIDDEN_COLS,
      ),
      right: matrixData(
        'attention update',
        'M',
        run.mixerOutput,
        'activation',
        TOKEN_ROWS,
        HIDDEN_COLS,
      ),
      result: matrixData(
        'mixer residual',
        'X_after_mixer',
        run.residualOutput,
        'result',
        TOKEN_ROWS,
        HIDDEN_COLS,
      ),
      details: elementwiseDetails(
        run.input,
        run.mixerOutput,
        run.residualOutput,
        'add',
      ),
      defaultCell: [3, 2],
    },
  });

  addFfnSteps(steps, groupId, groupTitle, layerIndex, run.ffn);
  return steps;
}

function sequenceParallelOperations(step) {
  const operations = step.visual.operations;
  if (operations.length <= 1) {
    return [step];
  }

  const orderedLabels = operations
    .map((operation) => operation.tabLabel)
    .join(' -> ');

  return operations.map((operation, index) => {
    const result = operation.result ?? operation.output;
    const rows = result.values.length;
    const cols = result.values[0].length;
    return {
      ...step,
      id: operation.id,
      shortTitle: operation.tabLabel,
      title: operation.title,
      paragraphs: [
        operation.caption,
        ...step.paragraphs,
      ],
      formula: operation.equation ?? operation.formula,
      sequenceNote:
        `These ${operations.length} branches use already-available inputs and may run in parallel in an optimized implementation. ` +
        `For learning, Continue visits them one at a time in this fixed order: ${orderedLabels}. ` +
        `This is operation ${index + 1} of ${operations.length}.`,
      callouts: [
        ['Serialized order', `${index + 1} of ${operations.length}`],
        ['Output shape', `${rows} x ${cols}`],
        [
          'Next',
          operations[index + 1]?.tabLabel ?? 'rejoin the forward path',
        ],
      ],
      visual: {
        ...step.visual,
        operations: [operation],
      },
    };
  });
}

function sequenceDeltaScan(step) {
  const sequence =
    'Each value head owns one state that evolves left-to-right across the whole sequence. The token controls inspect checkpoints in that one chain; they are not independent token states. At each checkpoint the order is decay -> predict -> correct -> write -> update -> read, then all o_t rows are assembled into O_core.';
  const phases = [
    {
      phase: 'chain',
      suffix: 'chain',
      shortTitle: '1. State chain overview',
      title: 'Carry one recurrent state through the token sequence',
      formula:
        'S_-1 --t0--> S_0 --t1--> S_1 --t2--> S_2 --t3--> S_3',
      paragraphs: [
        'A value head does not calculate four unrelated states. It starts from one initial state and mutates it once per token from left to right.',
        'Head 0 and head 1 have separate memory chains, but within either head S_t depends directly on S_(t-1). The controls below choose which checkpoint to inspect.',
        'An optimized prefill kernel can evaluate this recurrence in chunks, but that is an algebraic acceleration of the same causal state chain, not four independently initialized states.',
      ],
    },
    {
      phase: 'decay',
      suffix: 'decay',
      shortTitle: '2. Decay old state',
      title: 'Decay every old state cell by the retention factor exp(g)',
      formula: 'S_tilde = exp(g_t) x S_(t-1)',
      paragraphs: [
        'Start with the state produced by the previous token. The negative log-decay g becomes a retention factor exp(g) between zero and one.',
        'Multiply every state cell by the same per-token, per-head retention factor. Nothing new has been written yet.',
      ],
    },
    {
      phase: 'predict',
      suffix: 'predict',
      shortTitle: '3. Predict kS',
      title: 'Use the current key to read what the decayed state predicts',
      formula: 'memory = k_t S_tilde',
      paragraphs: [
        'Take the current key row and multiply it by the decayed key-by-value state.',
        'The resulting value-width vector is the memory prediction for this key.',
      ],
    },
    {
      phase: 'correct',
      suffix: 'correct',
      shortTitle: '4. Correct error',
      title: 'Compare the target value with memory and scale the correction by beta',
      formula: 'delta_t = beta_t x (v_t - memory)',
      paragraphs: [
        'Subtract the state prediction from the current value target coordinate by coordinate.',
        'Multiply the error by beta. A small beta makes a cautious update; a large beta writes most of the correction.',
      ],
    },
    {
      phase: 'write',
      suffix: 'write',
      shortTitle: '5. Write k^T delta',
      title: 'Turn the correction into a rank-one state write',
      formula: 'write = k_t^T delta_t',
      paragraphs: [
        'Transpose the key into a column and multiply it by the correction row.',
        'This outer product creates one update value for every key-by-value state cell.',
      ],
    },
    {
      phase: 'update',
      suffix: 'update',
      shortTitle: '6. Update state',
      title: 'Add the rank-one write to the decayed state',
      formula: 'S_t = S_tilde + write',
      paragraphs: [
        'The write matrix and decayed state have identical coordinates.',
        'Add them cell by cell. This completed state is carried to the next token.',
      ],
    },
    {
      phase: 'read',
      suffix: 'read',
      shortTitle: '7. Read qS',
      title: 'Use the query to read the newly updated state',
      formula: 'o_t = q_t S_t',
      paragraphs: [
        'The query multiplies the updated state after the current token has written its correction.',
        'The resulting value-width vector becomes this head output for the current token.',
      ],
    },
    {
      phase: 'assemble',
      suffix: 'assemble',
      shortTitle: '8. Assemble O_core',
      title: 'Stack every token read and concatenate the two value heads',
      formula:
        'O_core[t,:] = concat(o_t^(head 0), o_t^(head 1))',
      paragraphs: [
        'Repeating the recurrent update through t0, t1, t2, and t3 produces one o_t row per token for each value head.',
        'Stack rows down the time axis inside each head, then concatenate head 0 and head 1 across the feature axis. The resulting 4 x 4 tensor is O_core, the direct input to gated RMSNorm.',
      ],
    },
  ];

  return phases.map((phase, index) => ({
    ...step,
    id: `${step.id}-${phase.suffix}`,
    shortTitle: phase.shortTitle,
    title: phase.title,
    formula: phase.formula,
    paragraphs: phase.paragraphs,
    sequenceNote: sequence,
    callouts: [
      ['Scan action', `${index + 1} of ${phases.length}`],
      [
        'Scope',
        phase.phase === 'chain'
          ? 'one head across all tokens'
          : phase.phase === 'assemble'
            ? 'all tokens + both heads'
            : 'one checkpoint + one head',
      ],
      ['Next', phases[index + 1]?.shortTitle ?? 'gate the head output'],
    ],
    visual: {
      ...step.visual,
      phase: phase.phase,
    },
  }));
}

function sequenceSoftmax(step) {
  const { headIndex, totalHeads } = step.visual;
  const nextAfterNormalize =
    headIndex + 1 < totalHeads
      ? `causal mask for query head ${headIndex + 1}`
      : 'P x V context matmul for query head 0';
  return [
    {
      ...step,
      id: `${step.id}-mask`,
      shortTitle: `Head ${headIndex} causal mask`,
      title: `Mask future-token scores for query head ${headIndex}`,
      formula: `Masked_${headIndex}[t,j] = j > t ? -inf : Scores_${headIndex}[t,j]`,
      paragraphs: [
        `Process one query row from head ${headIndex} at a time. Scores to the right of the diagonal point into the future and must not influence the current token.`,
        'Replacing those entries with negative infinity guarantees their softmax probability becomes exactly zero.',
      ],
      sequenceNote:
        `Head ${headIndex} masking must happen before head ${headIndex} softmax. The tutorial completes both actions before moving to the next query head.`,
      callouts: [
        ['Attention action', '1 of 2'],
        ['Query head', `${headIndex} of ${totalHeads - 1}`],
        ['Future entries', '-inf'],
      ],
      visual: {
        ...step.visual,
        phase: 'mask',
      },
    },
    {
      ...step,
      id: `${step.id}-normalize`,
      shortTitle: `Head ${headIndex} row softmax`,
      title: `Normalize the unmasked score row for query head ${headIndex}`,
      formula:
        `P_${headIndex}[t,j] = exp(Masked_${headIndex}[t,j] - row_max) / sum_k exp(Masked_${headIndex}[t,k] - row_max)`,
      paragraphs: [
        'Subtract the largest finite score for numerical stability, exponentiate each surviving entry, and add those exponentials.',
        `Divide each exponential by that sum. Head ${headIndex}'s row is now positive, sums to one, and assigns zero probability to the future.`,
      ],
      sequenceNote:
        `This completes score normalization for query head ${headIndex}. Continue moves to ${nextAfterNormalize}.`,
      callouts: [
        ['Attention action', '2 of 2'],
        ['Probability row sum', '1.000'],
        ['Next', nextAfterNormalize],
      ],
      visual: {
        ...step.visual,
        phase: 'normalize',
      },
    },
  ];
}

function sequenceTutorialSteps(steps) {
  return steps.flatMap((step) => {
    if (step.visual.kind === 'delta' && !step.visual.phase) {
      return sequenceDeltaScan(step);
    }
    if (step.visual.kind === 'softmax' && !step.visual.phase) {
      return sequenceSoftmax(step);
    }
    if (
      (step.visual.kind === 'matmul' ||
        step.visual.kind === 'transform') &&
      step.visual.operations.length > 1
    ) {
      return sequenceParallelOperations(step);
    }
    return [step];
  });
}

function buildTutorial() {
  const steps = [
    {
      id: 'tokens',
      groupId: 'orientation',
      groupTitle: 'Orientation',
      shortTitle: 'Sample tokens',
      eyebrow: 'Input / tokenizer output',
      title: 'Start with four sample token IDs',
      architectureTarget: 'tokens',
      paragraphs: [
        'The tokenizer has already converted text pieces into integers. The neural network never receives the strings themselves.',
        'The teaching vocabulary has eight entries so every row stays visible. Qwen3.5-4B uses 248,320 vocabulary rows.',
      ],
      formula: 'text pieces -> integer token IDs',
      callouts: [
        ['Prompt', 'A tiny robot learns'],
        ['Sequence length', '4'],
        ['Real vocabulary', '248,320'],
      ],
      visual: {
        kind: 'tokens',
        tokens: TOKENS,
        tokenIds: TOKEN_IDS,
        vocabulary: VOCABULARY,
      },
    },
    {
      id: 'embedding',
      groupId: 'orientation',
      groupTitle: 'Orientation',
      shortTitle: 'Embedding lookup',
      eyebrow: 'Input / tied token embedding',
      title: 'Copy one learned embedding row for each token ID',
      architectureTarget: 'embedding',
      paragraphs: [
        'Embedding is a row lookup, not a matrix multiplication. Token ID 3 simply selects row 3 from the blue learned table.',
        'The copied green rows form the first residual stream. Qwen3.5 does not add an absolute position embedding here.',
      ],
      formula: 'X[t,:] = Embedding[token_id[t],:]',
      callouts: [
        ['Mini table', '8 x 8'],
        ['Result', '4 x 8'],
        ['Absolute position add', 'none'],
      ],
      visual: {
        kind: 'lookup',
        tokens: TOKENS,
        tokenIds: TOKEN_IDS,
        embeddingTable: matrixData(
          'token embedding table',
          'E',
          MODEL_RUN.embeddingTable,
          'weight',
          VOCABULARY.map((token, index) => `${index}:${token}`),
          HIDDEN_COLS,
        ),
        result: matrixData(
          'initial residual stream',
          'X0',
          MODEL_RUN.embedded,
          'result',
          TOKEN_ROWS,
          HIDDEN_COLS,
        ),
      },
    },
    {
      id: 'quartet-overview',
      groupId: 'orientation',
      groupTitle: 'Orientation',
      shortTitle: '3:1 layer pattern',
      eyebrow: 'Qwen3.5-4B / hidden layout',
      title: 'Follow one four-layer Qwen3.5 quartet from top to bottom',
      architectureTarget: 'cycle',
      paragraphs: [
        'The dense 4B checkpoint has 32 decoder layers arranged as eight repeats of three Gated DeltaNet layers followed by one gated full-attention layer.',
        'This page expands one quartet. Widths are reduced, but operation order, head-sharing ratios, residuals, dense SwiGLU FFNs, and tied output weights are preserved.',
      ],
      formula:
        '8 x [DeltaNet -> DeltaNet -> DeltaNet -> Full attention]',
      callouts: [
        ['Teaching layers', '0, 1, 2, 3'],
        ['Real repeats', '8'],
        ['MoE routing', 'none in 4B'],
      ],
      visual: { kind: 'overview' },
    },
  ];

  steps.push(...buildDeltaSteps(MODEL_RUN.deltaLayers[0]));
  steps.push(
    buildDeltaRepeatStep(
      MODEL_RUN.deltaLayers[0],
      MODEL_RUN.deltaLayers[1],
      MODEL_RUN.deltaLayers[2],
    ),
  );
  steps.push(...buildAttentionSteps(MODEL_RUN.attentionLayer));

  steps.push({
    id: 'final-norm',
    groupId: 'output',
    groupTitle: 'Output head',
    shortTitle: 'Final RMSNorm',
    eyebrow: 'After the quartet / model output',
    title: 'Normalize the completed residual stream one last time',
    architectureTarget: 'final-norm',
    paragraphs: [
      'A full 4B run would repeat seven more quartets before this point. The teaching run stops after one quartet to keep every value inspectable.',
      'The final RMSNorm prepares hidden states for vocabulary projection without mixing tokens.',
    ],
    formula: 'H = RMSNorm(X_final)',
    callouts: [
      ['Teaching depth', '4 layers'],
      ['Real depth', '32 layers'],
      ['Output shape', '4 x 8'],
    ],
    visual: {
      kind: 'transform',
      operations: [
        rmsOperation(
          'final-rms',
          'Final model RMSNorm',
          MODEL_RUN.hidden,
          MODEL_RUN.finalNormalized,
          MODEL_RUN.finalGamma,
        ),
      ],
    },
  });

  steps.push({
    id: 'lm-head',
    groupId: 'output',
    groupTitle: 'Output head',
    shortTitle: 'Tied LM-head matmul',
    eyebrow: 'Vocabulary projection',
    title: 'Multiply hidden rows by the transposed token embedding table',
    architectureTarget: 'lm-head',
    paragraphs: [
      'Qwen3.5-4B ties output weights to the input embedding table. The same blue values are viewed transposed.',
      'Each green logit is a dot product between one final hidden row and one vocabulary embedding row.',
    ],
    formula: 'Logits = H E^T',
    callouts: [
      ['Mini matmul', '(4 x 8) x (8 x 8)'],
      ['Real output width', '248,320'],
      ['Weights tied', 'E^T'],
    ],
    visual: {
      kind: 'matmul',
      operations: [
        makeMatmulOperation({
          id: 'lm-head-projection',
          tabLabel: 'Vocabulary logits',
          title: 'Tied output projection',
          equation: 'Logits = H x E^T',
          caption:
            'Choose any green token logit. Its products compare a hidden row with one blue vocabulary embedding.',
          left: matrixData(
            'final hidden states',
            'H',
            MODEL_RUN.finalNormalized,
            'activation',
            TOKEN_ROWS,
            HIDDEN_COLS,
          ),
          right: matrixData(
            'transposed tied embedding',
            'E^T',
            transpose(MODEL_RUN.embeddingTable),
            'weight',
            HIDDEN_COLS,
            VOCABULARY,
          ),
          resultName: 'vocabulary logits',
          resultSymbol: 'Logits',
          resultRowLabels: TOKEN_ROWS,
          resultColLabels: VOCABULARY,
          defaultCell: [3, MODEL_RUN.nextTokenId],
        }),
      ],
    },
  });

  steps.push({
    id: 'next-token',
    groupId: 'output',
    groupTitle: 'Output head',
    shortTitle: 'Next-token distribution',
    eyebrow: 'Autoregressive output',
    title: `Normalize the last-row logits and select "${MODEL_RUN.nextToken}"`,
    architectureTarget: 'next-token',
    paragraphs: [
      'Softmax turns the last token row into a probability distribution. Greedy decoding picks the largest value; sampling would draw from the full distribution.',
      'The fixed toy embedding row for "fast" is intentionally chosen to make this example read naturally after "A tiny robot learns." It still demonstrates tied-weight arithmetic rather than pretrained language ability.',
    ],
    formula: 'p(next_token) = softmax(Logits[last,:])',
    callouts: [
      ['Selected token', MODEL_RUN.nextToken],
      ['Decoder shown', 'greedy argmax'],
      ['Toy target', 'intentionally "fast"'],
    ],
    visual: {
      kind: 'prediction',
      tokens: VOCABULARY,
      logits: matrixData(
        'vocabulary logits',
        'Logits',
        MODEL_RUN.logits,
        'activation',
        TOKEN_ROWS,
        VOCABULARY,
      ),
      probabilities: matrixData(
        'token probabilities',
        'P_vocab',
        MODEL_RUN.probabilities,
        'probability',
        TOKEN_ROWS,
        VOCABULARY,
      ),
      nextToken: MODEL_RUN.nextToken,
      nextTokenId: MODEL_RUN.nextTokenId,
    },
  });

  const groupOrder = [
    'orientation',
    'delta-0',
    'delta-repeat',
    'full-attention',
    'output',
  ];
  const subtitles = {
    orientation: 'Tokens, embeddings, and the 3:1 map',
    'delta-0': 'First recurrent token mixer + dense FFN',
    'delta-repeat': 'Same graph twice, with new weights and states',
    'full-attention': 'Grouped-query all-pairs mixing + dense FFN',
    output: 'Final norm, tied logits, and next token',
  };
  const sequencedSteps = sequenceTutorialSteps(steps);
  const groups = groupOrder.map((groupId) => {
    const matching = sequencedSteps.filter((item) => item.groupId === groupId);
    return {
      id: groupId,
      title: matching[0].groupTitle,
      subtitle: subtitles[groupId],
      stepIds: matching.map((item) => item.id),
    };
  });

  return {
    steps: sequencedSteps,
    groups,
    realConfig: REAL_CONFIG,
    miniConfig: MINI_CONFIG,
    tokens: TOKENS,
    vocabulary: VOCABULARY,
  };
}

export const QWEN_TUTORIAL = buildTutorial();
