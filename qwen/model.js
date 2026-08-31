import {
  add,
  cloneMatrix,
  concatColumns,
  formatNumber,
  l2NormalizeRows,
  mapMatrix,
  matmul,
  rmsNorm,
  scale,
  seededMatrix,
  seededVector,
  sigmoid,
  silu,
  sliceColumns,
  softmaxRows,
  softplus,
  transpose,
  zipMatrix,
  zeros,
} from './math.js';

export const REAL_CONFIG = {
  model: 'Qwen3.5-4B',
  hiddenSize: 2560,
  layers: 32,
  pattern: '8 x (3 DeltaNet + 1 full attention)',
  vocabulary: 248320,
  context: 262144,
  deltaHeads: '16 Q/K heads, 32 V heads',
  deltaHeadDim: 128,
  attentionQueryHeads: 16,
  attentionKvHeads: 4,
  attentionHeads: '16 Q heads, 4 K/V heads',
  attentionHeadDim: 256,
  rotaryDimensions: 64,
  ffnSize: 9216,
  convKernel: 4,
};

export const MINI_CONFIG = {
  model: 'Qwen3.5 Mini (teaching scale)',
  hiddenSize: 8,
  layers: 4,
  pattern: '1 x (3 DeltaNet + 1 full attention)',
  vocabulary: 8,
  context: 4,
  deltaHeads: '1 Q/K head, 2 V heads',
  deltaHeadDim: 2,
  attentionQueryHeads: 4,
  attentionKvHeads: 1,
  attentionHeads: '4 Q heads, 1 K/V head',
  attentionHeadDim: 2,
  rotaryDimensions: 2,
  ffnSize: 12,
  convKernel: 4,
};

export const TOKENS = ['A', 'tiny', 'robot', 'learns'];
export const TOKEN_IDS = [1, 2, 3, 4];
export const VOCABULARY = [
  '<bos>',
  'A',
  'tiny',
  'robot',
  'learns',
  'fast',
  '.',
  '<eos>',
];
export const TOKEN_ROWS = TOKENS.map((token, index) => `t${index}:${token}`);
export const HIDDEN_COLS = Array.from(
  { length: MINI_CONFIG.hiddenSize },
  (_, index) => `c${index}`,
);
export const FFN_COLS = Array.from(
  { length: MINI_CONFIG.ffnSize },
  (_, index) => `m${index}`,
);

export function matrixData(
  name,
  symbol,
  values,
  role,
  rowLabels,
  colLabels,
) {
  return { name, symbol, values, role, rowLabels, colLabels };
}

export function rmsNormDetails(input, gamma) {
  return input.map((row) => {
    const meanSquare =
      row.reduce((sum, value) => sum + value * value, 0) / row.length;
    const rms = Math.sqrt(meanSquare + 1e-6);
    return row.map(
      (value, colIndex) =>
        `${formatNumber(value)} / sqrt(mean(row^2)=${formatNumber(meanSquare)}) x gamma[${colIndex}]=${formatNumber(gamma[colIndex])} = ${formatNumber(value / rms * gamma[colIndex])}`,
    );
  });
}

export function elementwiseDetails(left, right, result, operator) {
  return result.map((row, rowIndex) =>
    row.map((_, colIndex) => {
      const leftValue = left[rowIndex][colIndex];
      const rightValue = right[rowIndex][colIndex];
      if (operator === 'add') {
        return `${formatNumber(leftValue)} + ${formatNumber(rightValue)} = ${formatNumber(result[rowIndex][colIndex])}`;
      }
      if (operator === 'gate') {
        return `${formatNumber(leftValue)} x sigmoid(${formatNumber(rightValue)}) = ${formatNumber(result[rowIndex][colIndex])}`;
      }
      return `SiLU(${formatNumber(leftValue)}) x ${formatNumber(rightValue)} = ${formatNumber(result[rowIndex][colIndex])}`;
    }),
  );
}

export function makeMatmulOperation(args) {
  const termScale = args.termScale ?? 1;
  const rawResult = matmul(args.left.values, args.right.values);
  const resultValues =
    args.resultValues ?? (termScale === 1 ? rawResult : scale(rawResult, termScale));
  return {
    id: args.id,
    tabLabel: args.tabLabel,
    title: args.title,
    equation: args.equation,
    caption: args.caption,
    left: args.left,
    right: args.right,
    result: matrixData(
      args.resultName,
      args.resultSymbol,
      resultValues,
      args.resultRole ?? 'result',
      args.resultRowLabels,
      args.resultColLabels,
    ),
    defaultCell: args.defaultCell,
    termScale,
    derived: args.derived,
  };
}

function runFfn(input, seed) {
  const normGamma = seededVector(MINI_CONFIG.hiddenSize, seed + 1, 0.06, 1);
  const normalized = rmsNorm(input, normGamma);
  const gateWeight = seededMatrix(
    MINI_CONFIG.hiddenSize,
    MINI_CONFIG.ffnSize,
    seed + 2,
    0.22,
  );
  const upWeight = seededMatrix(
    MINI_CONFIG.hiddenSize,
    MINI_CONFIG.ffnSize,
    seed + 3,
    0.22,
  );
  const gate = matmul(normalized, gateWeight);
  const up = matmul(normalized, upWeight);
  const activated = zipMatrix(gate, up, (gateValue, upValue) =>
    silu(gateValue) * upValue,
  );
  const downWeight = seededMatrix(
    MINI_CONFIG.ffnSize,
    MINI_CONFIG.hiddenSize,
    seed + 4,
    0.18,
  );
  const down = matmul(activated, downWeight);
  const output = add(input, down);

  return {
    input,
    normalized,
    normGamma,
    gateWeight,
    upWeight,
    gate,
    up,
    activated,
    downWeight,
    down,
    output,
  };
}

function buildConvKernels(channels, seed) {
  const noise = seededMatrix(
    channels,
    MINI_CONFIG.convKernel,
    seed,
    0.08,
  );
  return noise.map((row) =>
    row.map((value, kernelIndex) => {
      const base = [0.08, 0.14, 0.24, 0.62][kernelIndex];
      return base + value;
    }),
  );
}

function causalDepthwiseConv(input, kernels) {
  const output = zeros(input.length, input[0].length);
  const traces = Array.from({ length: input.length }, () => []);

  for (let timeIndex = 0; timeIndex < input.length; timeIndex += 1) {
    for (
      let channelIndex = 0;
      channelIndex < input[0].length;
      channelIndex += 1
    ) {
      const terms = [];
      let sum = 0;
      for (
        let kernelIndex = 0;
        kernelIndex < MINI_CONFIG.convKernel;
        kernelIndex += 1
      ) {
        const inputTime =
          timeIndex - (MINI_CONFIG.convKernel - 1 - kernelIndex);
        const inputValue =
          inputTime >= 0 ? input[inputTime][channelIndex] : 0;
        const kernelValue = kernels[channelIndex][kernelIndex];
        const product = inputValue * kernelValue;
        terms.push({
          inputTime: inputTime >= 0 ? inputTime : null,
          inputValue,
          kernelIndex,
          kernelValue,
          product,
        });
        sum += product;
      }
      const activated = silu(sum);
      output[timeIndex][channelIndex] = activated;
      traces[timeIndex][channelIndex] = { terms, sum, activated };
    }
  }

  return { output, traces };
}

function runDeltaLayer(input, layerIndex) {
  const seed = 1000 + layerIndex * 100;
  const normGamma = seededVector(MINI_CONFIG.hiddenSize, seed + 1, 0.06, 1);
  const normalized = rmsNorm(input, normGamma);

  const keyDim = 2;
  const valueDim = 4;
  const qkvWidth = keyDim * 2 + valueDim;
  const qkvWeight = seededMatrix(
    MINI_CONFIG.hiddenSize,
    qkvWidth,
    seed + 2,
    0.26,
  );
  const qkvProjected = matmul(normalized, qkvWeight);

  const zWeight = seededMatrix(
    MINI_CONFIG.hiddenSize,
    valueDim,
    seed + 3,
    0.24,
  );
  const z = matmul(normalized, zWeight);

  const betaWeight = seededMatrix(
    MINI_CONFIG.hiddenSize,
    2,
    seed + 4,
    0.22,
  );
  const betaLogits = matmul(normalized, betaWeight);
  const beta = mapMatrix(betaLogits, sigmoid);

  const decayWeight = seededMatrix(
    MINI_CONFIG.hiddenSize,
    2,
    seed + 5,
    0.18,
  );
  const decayLogits = matmul(normalized, decayWeight);
  const decayRates = seededVector(2, seed + 6, 0.08, 0.18);
  const dtBias = seededVector(2, seed + 7, 0.08, 0.25);
  const decay = decayLogits.map((row) =>
    row.map(
      (value, headIndex) =>
        -decayRates[headIndex] *
        softplus(value + dtBias[headIndex]),
    ),
  );

  const convKernels = buildConvKernels(qkvWidth, seed + 8);
  const { output: convolvedQkv, traces: convTraces } =
    causalDepthwiseConv(qkvProjected, convKernels);
  const queryBase = l2NormalizeRows(
    sliceColumns(convolvedQkv, 0, keyDim),
  );
  const keyBase = l2NormalizeRows(
    sliceColumns(convolvedQkv, keyDim, keyDim * 2),
  );
  const queryScaled = scale(queryBase, 1 / Math.sqrt(keyDim));
  const values = sliceColumns(convolvedQkv, keyDim * 2, qkvWidth);

  const deltaTraces = [];
  const perHeadOutput = [zeros(input.length, 2), zeros(input.length, 2)];

  for (let headIndex = 0; headIndex < 2; headIndex += 1) {
    const valueLabels = [`v${headIndex * 2}`, `v${headIndex * 2 + 1}`];
    let state = zeros(2, 2);
    for (let tokenIndex = 0; tokenIndex < input.length; tokenIndex += 1) {
      const q = [queryScaled[tokenIndex]];
      const k = [keyBase[tokenIndex]];
      const v = [
        values[tokenIndex].slice(headIndex * 2, headIndex * 2 + 2),
      ];
      const stateBefore = cloneMatrix(state);
      const decayFactor = Math.exp(decay[tokenIndex][headIndex]);
      const stateDecayed = scale(stateBefore, decayFactor);
      const memory = matmul(k, stateDecayed);
      const correction = zipMatrix(
        v,
        memory,
        (value, prediction) =>
          (value - prediction) * beta[tokenIndex][headIndex],
      );
      const write = matmul(transpose(k), correction);
      state = add(stateDecayed, write);
      const read = matmul(q, state);
      perHeadOutput[headIndex][tokenIndex] = [...read[0]];

      deltaTraces.push({
        tokenIndex,
        headIndex,
        decay: decayFactor,
        beta: beta[tokenIndex][headIndex],
        q: matrixData(
          'scaled query',
          'q_t',
          q,
          'activation',
          [`t${tokenIndex}`],
          ['k0', 'k1'],
        ),
        k: matrixData(
          'normalized key',
          'k_t',
          k,
          'activation',
          [`t${tokenIndex}`],
          ['k0', 'k1'],
        ),
        v: matrixData(
          'value target',
          'v_t',
          v,
          'activation',
          [`t${tokenIndex}`],
          valueLabels,
        ),
        stateBefore: matrixData(
          'state before decay',
          'S_(t-1)',
          stateBefore,
          'state',
          ['k0', 'k1'],
          valueLabels,
        ),
        stateDecayed: matrixData(
          'decayed state',
          'exp(g_t) S_(t-1)',
          stateDecayed,
          'state',
          ['k0', 'k1'],
          valueLabels,
        ),
        memory: matrixData(
          'state prediction',
          'k_t S',
          memory,
          'result',
          [`t${tokenIndex}`],
          valueLabels,
        ),
        correction: matrixData(
          'delta correction',
          'delta_t',
          correction,
          'gate',
          [`t${tokenIndex}`],
          valueLabels,
        ),
        write: matrixData(
          'rank-one write',
          'k_t^T delta_t',
          write,
          'result',
          ['k0', 'k1'],
          valueLabels,
        ),
        stateAfter: matrixData(
          'updated state',
          'S_t',
          state,
          'state',
          ['k0', 'k1'],
          valueLabels,
        ),
        read: matrixData(
          'head output',
          'q_t S_t',
          read,
          'result',
          [`t${tokenIndex}`],
          valueLabels,
        ),
      });
    }
  }

  const coreOutput = concatColumns(perHeadOutput[0], perHeadOutput[1]);
  const gatedGamma = seededVector(2, seed + 9, 0.06, 1);
  const gatedOutput = coreOutput.map((row, tokenIndex) => {
    const result = [];
    for (let headIndex = 0; headIndex < 2; headIndex += 1) {
      const headValues = row.slice(headIndex * 2, headIndex * 2 + 2);
      const meanSquare =
        headValues.reduce((sum, value) => sum + value * value, 0) /
        headValues.length;
      const inverseRms = 1 / Math.sqrt(meanSquare + 1e-6);
      const zValues = z[tokenIndex].slice(
        headIndex * 2,
        headIndex * 2 + 2,
      );
      result.push(
        ...headValues.map(
          (value, featureIndex) =>
            value *
            inverseRms *
            gatedGamma[featureIndex] *
            silu(zValues[featureIndex]),
        ),
      );
    }
    return result;
  });

  const outWeight = seededMatrix(
    valueDim,
    MINI_CONFIG.hiddenSize,
    seed + 10,
    0.24,
  );
  const mixerOutput = matmul(gatedOutput, outWeight);
  const residualOutput = add(input, mixerOutput);
  const ffn = runFfn(residualOutput, seed + 20);

  return {
    layerIndex,
    input,
    normalized,
    normGamma,
    qkvWeight,
    qkvProjected,
    zWeight,
    z,
    betaWeight,
    betaLogits,
    beta,
    decayWeight,
    decayLogits,
    decay,
    convKernels,
    convolvedQkv,
    convTraces,
    deltaTraces,
    coreOutput,
    gatedOutput,
    gatedGamma,
    outWeight,
    mixerOutput,
    residualOutput,
    ffn,
  };
}

function applyPartialRope(matrix, headIndex) {
  return matrix.map((row, tokenIndex) => {
    const angle = tokenIndex * (0.42 + headIndex * 0.09);
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    return [
      row[0] * cos - row[1] * sin,
      row[0] * sin + row[1] * cos,
      ...row.slice(2),
    ];
  });
}

function normalizeHead(matrix, gamma) {
  return rmsNorm(matrix, gamma);
}

function runAttentionLayer(input, layerIndex) {
  const seed = 2000 + layerIndex * 100;
  const numQHeads = MINI_CONFIG.attentionQueryHeads;
  const headDim = MINI_CONFIG.attentionHeadDim;
  const queryWidth = numQHeads * headDim;
  const normGamma = seededVector(MINI_CONFIG.hiddenSize, seed + 1, 0.06, 1);
  const normalized = rmsNorm(input, normGamma);

  const qGateWeight = seededMatrix(
    MINI_CONFIG.hiddenSize,
    queryWidth * 2,
    seed + 2,
    0.23,
  );
  const qGateProjected = matmul(normalized, qGateWeight);
  const qRaw = sliceColumns(qGateProjected, 0, queryWidth);
  const outputGate = sliceColumns(
    qGateProjected,
    queryWidth,
    queryWidth * 2,
  );
  const kWeight = seededMatrix(
    MINI_CONFIG.hiddenSize,
    headDim,
    seed + 3,
    0.24,
  );
  const kRaw = matmul(normalized, kWeight);
  const vWeight = seededMatrix(
    MINI_CONFIG.hiddenSize,
    headDim,
    seed + 4,
    0.24,
  );
  const vRaw = matmul(normalized, vWeight);

  const qNormGamma = seededVector(headDim, seed + 5, 0.05, 1);
  const kNormGamma = seededVector(headDim, seed + 6, 0.05, 1);
  const qBeforeRope = Array.from({ length: numQHeads }, (_, headIndex) =>
    normalizeHead(
      sliceColumns(
        qRaw,
        headIndex * headDim,
        (headIndex + 1) * headDim,
      ),
      qNormGamma,
    ),
  );
  const sharedKey = normalizeHead(kRaw, kNormGamma);
  const qAfterRope = qBeforeRope.map((head, headIndex) =>
    applyPartialRope(head, headIndex),
  );
  const sharedKeyAfterRope = applyPartialRope(sharedKey, 0);
  const kBeforeRope = Array.from({ length: numQHeads }, () => sharedKey);
  const kAfterRope = Array.from(
    { length: numQHeads },
    () => sharedKeyAfterRope,
  );
  const vHeads = Array.from({ length: numQHeads }, () => vRaw);

  const scores = qAfterRope.map((queryHead, headIndex) =>
    scale(
      matmul(queryHead, transpose(kAfterRope[headIndex])),
      1 / Math.sqrt(headDim),
    ),
  );
  const maskedScores = scores.map((headScores) =>
    headScores.map((row, rowIndex) =>
      row.map((value, colIndex) =>
        colIndex > rowIndex ? Number.NEGATIVE_INFINITY : value,
      ),
    ),
  );
  const probabilities = maskedScores.map(softmaxRows);
  const contextHeads = probabilities.map(
    (headProbabilities, headIndex) =>
      matmul(headProbabilities, vHeads[headIndex]),
  );
  const context = concatColumns(...contextHeads);
  const gatedContext = zipMatrix(
    context,
    outputGate,
    (value, gate) => value * sigmoid(gate),
  );
  const outWeight = seededMatrix(
    queryWidth,
    MINI_CONFIG.hiddenSize,
    seed + 7,
    0.23,
  );
  const mixerOutput = matmul(gatedContext, outWeight);
  const residualOutput = add(input, mixerOutput);
  const ffn = runFfn(residualOutput, seed + 20);

  return {
    layerIndex,
    input,
    normalized,
    normGamma,
    qGateWeight,
    qGateProjected,
    qRaw,
    outputGate,
    kWeight,
    kRaw,
    vWeight,
    vRaw,
    qNormGamma,
    kNormGamma,
    qBeforeRope,
    qAfterRope,
    kBeforeRope,
    kAfterRope,
    vHeads,
    scores,
    maskedScores,
    probabilities,
    contextHeads,
    context,
    gatedContext,
    outWeight,
    mixerOutput,
    residualOutput,
    ffn,
  };
}

function buildModelRun() {
  const embeddingTable = seededMatrix(
    VOCABULARY.length,
    MINI_CONFIG.hiddenSize,
    77,
    0.72,
  );
  const embedded = TOKEN_IDS.map((tokenId) => [...embeddingTable[tokenId]]);
  const deltaLayers = [];
  let hidden = embedded;
  for (let layerIndex = 0; layerIndex < 3; layerIndex += 1) {
    const layer = runDeltaLayer(hidden, layerIndex);
    deltaLayers.push(layer);
    hidden = layer.ffn.output;
  }
  const attentionLayer = runAttentionLayer(hidden, 3);
  hidden = attentionLayer.ffn.output;

  const finalGamma = seededVector(
    MINI_CONFIG.hiddenSize,
    3001,
    0.06,
    1,
  );
  const finalNormalized = rmsNorm(hidden, finalGamma);
  const logits = matmul(finalNormalized, transpose(embeddingTable));
  const probabilities = softmaxRows(logits);
  const finalProbabilityRow = probabilities[probabilities.length - 1];
  const nextTokenId = finalProbabilityRow.reduce(
    (bestIndex, value, index, row) =>
      value > row[bestIndex] ? index : bestIndex,
    0,
  );

  return {
    embeddingTable,
    embedded,
    deltaLayers,
    attentionLayer,
    hidden,
    finalGamma,
    finalNormalized,
    logits,
    probabilities,
    nextTokenId,
    nextToken: VOCABULARY[nextTokenId],
  };
}

export const MODEL_RUN = buildModelRun();
