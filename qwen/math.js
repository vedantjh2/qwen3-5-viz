export function createRng(seed) {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

export function seededMatrix(rows, cols, seed, scale, center = 0) {
  const random = createRng(seed);
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => center + (random() * 2 - 1) * scale),
  );
}

export function seededVector(length, seed, scale, center = 0) {
  return seededMatrix(1, length, seed, scale, center)[0];
}

export function zeros(rows, cols) {
  return Array.from({ length: rows }, () => Array(cols).fill(0));
}

export function cloneMatrix(matrix) {
  return matrix.map((row) => [...row]);
}

export function mapMatrix(matrix, fn) {
  return matrix.map((row, rowIndex) =>
    row.map((value, colIndex) => fn(value, rowIndex, colIndex)),
  );
}

export function zipMatrix(left, right, fn) {
  return left.map((row, rowIndex) =>
    row.map((value, colIndex) => fn(value, right[rowIndex][colIndex], rowIndex, colIndex)),
  );
}

export function transpose(matrix) {
  return matrix[0].map((_, colIndex) => matrix.map((row) => row[colIndex]));
}

export function matmul(left, right) {
  if (left[0].length !== right.length) {
    throw new Error(
      `Invalid matmul: ${left.length}x${left[0].length} by ${right.length}x${right[0].length}`,
    );
  }

  return left.map((row) =>
    right[0].map((_, colIndex) =>
      row.reduce(
        (sum, value, innerIndex) => sum + value * right[innerIndex][colIndex],
        0,
      ),
    ),
  );
}

export function add(left, right) {
  return zipMatrix(left, right, (leftValue, rightValue) => leftValue + rightValue);
}

export function multiply(left, right) {
  return zipMatrix(left, right, (leftValue, rightValue) => leftValue * rightValue);
}

export function scale(matrix, amount) {
  return mapMatrix(matrix, (value) => value * amount);
}

export function sliceColumns(matrix, start, end) {
  return matrix.map((row) => row.slice(start, end));
}

export function concatColumns(...matrices) {
  return matrices[0].map((_, rowIndex) =>
    matrices.flatMap((matrix) => matrix[rowIndex]),
  );
}

export function silu(value) {
  return value / (1 + Math.exp(-value));
}

export function sigmoid(value) {
  return 1 / (1 + Math.exp(-value));
}

export function softplus(value) {
  if (value > 20) {
    return value;
  }
  return Math.log1p(Math.exp(value));
}

export function rmsNorm(matrix, gamma, epsilon = 1e-6) {
  return matrix.map((row) => {
    const meanSquare =
      row.reduce((sum, value) => sum + value * value, 0) / row.length;
    const inverseRms = 1 / Math.sqrt(meanSquare + epsilon);
    return row.map(
      (value, colIndex) => value * inverseRms * gamma[colIndex],
    );
  });
}

export function l2NormalizeRows(matrix) {
  return matrix.map((row) => {
    const norm = Math.sqrt(
      row.reduce((sum, value) => sum + value * value, 0) + 1e-6,
    );
    return row.map((value) => value / norm);
  });
}

export function softmaxRows(matrix) {
  return matrix.map((row) => {
    const finiteValues = row.filter(Number.isFinite);
    const maxValue = Math.max(...finiteValues);
    const exponentials = row.map((value) =>
      Number.isFinite(value) ? Math.exp(value - maxValue) : 0,
    );
    const denominator = exponentials.reduce((sum, value) => sum + value, 0);
    return exponentials.map((value) => value / denominator);
  });
}

export function formatNumber(value, digits = 3) {
  if (value === Number.NEGATIVE_INFINITY) {
    return '-inf';
  }
  const normalized = Math.abs(value) < 0.0005 ? 0 : value;
  return normalized.toFixed(digits);
}
