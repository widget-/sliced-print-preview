/**
 * Compute the inverse of a 4×4 column-major matrix in-place.
 * Returns the inverse, or the identity matrix if the input is singular.
 * Ported from glm::inverse (MIT license compatible).
 */
export function mat4Inverse(m: Float32Array, out: Float32Array): void {
  const a = m;
  const o = out;

  const A2323 = a[10] * a[15] - a[11] * a[14];
  const A1323 = a[9] * a[15] - a[11] * a[13];
  const A1223 = a[9] * a[14] - a[10] * a[13];
  const A0323 = a[8] * a[15] - a[11] * a[12];
  const A0223 = a[8] * a[14] - a[10] * a[12];
  const A0123 = a[8] * a[13] - a[9] * a[12];
  const A2313 = a[6] * a[15] - a[7] * a[14];
  const A1313 = a[5] * a[15] - a[7] * a[13];
  const A1213 = a[5] * a[14] - a[6] * a[13];
  const A2312 = a[6] * a[11] - a[7] * a[10];
  const A1312 = a[5] * a[11] - a[7] * a[9];
  const A1212 = a[5] * a[10] - a[6] * a[9];
  const A0313 = a[4] * a[15] - a[7] * a[12];
  const A0213 = a[4] * a[14] - a[6] * a[12];
  const A0312 = a[4] * a[11] - a[7] * a[8];
  const A0212 = a[4] * a[10] - a[6] * a[8];
  const A0113 = a[4] * a[13] - a[5] * a[12];
  const A0112 = a[4] * a[9] - a[5] * a[8];

  let det = a[0] * (a[5] * A2323 - a[6] * A1323 + a[7] * A1223)
          - a[1] * (a[4] * A2323 - a[6] * A0323 + a[7] * A0223)
          + a[2] * (a[4] * A1323 - a[5] * A0323 + a[7] * A0123)
          - a[3] * (a[4] * A1223 - a[5] * A0223 + a[6] * A0123);

  if (Math.abs(det) < 1e-15) {
    // Singular — return identity
    o.fill(0); o[0] = o[5] = o[10] = o[15] = 1;
    return;
  }
  det = 1.0 / det;

  o[0] = det *  (a[5] * A2323 - a[6] * A1323 + a[7] * A1223);
  o[1] = det * -(a[1] * A2323 - a[2] * A1323 + a[3] * A1223);
  o[2] = det *  (a[1] * A2313 - a[2] * A1313 + a[3] * A1213);
  o[3] = det * -(a[1] * A2312 - a[2] * A1312 + a[3] * A1212);
  o[4] = det * -(a[4] * A2323 - a[6] * A0323 + a[7] * A0223);
  o[5] = det *  (a[0] * A2323 - a[2] * A0323 + a[3] * A0223);
  o[6] = det * -(a[0] * A2313 - a[2] * A0313 + a[3] * A0213);
  o[7] = det *  (a[0] * A2312 - a[2] * A0312 + a[3] * A0212);
  o[8] = det *  (a[4] * A1323 - a[5] * A0323 + a[7] * A0123);
  o[9] = det * -(a[0] * A1323 - a[1] * A0323 + a[3] * A0123);
  o[10] = det * (a[0] * A1313 - a[1] * A0313 + a[3] * A0113);
  o[11] = det * -(a[0] * A1312 - a[1] * A0312 + a[3] * A0112);
  o[12] = det * -(a[4] * A1223 - a[5] * A0223 + a[6] * A0123);
  o[13] = det *  (a[0] * A1223 - a[1] * A0223 + a[2] * A0123);
  o[14] = det * -(a[0] * A1213 - a[1] * A0213 + a[2] * A0113);
  o[15] = det *  (a[0] * A1212 - a[1] * A0212 + a[2] * A0112);
}
