export interface BodyGeometry {
  /** Interleaved float32: [px, py, pz, nx, ny, nz] per vertex, 24 bytes/vertex. */
  interleaved: Float32Array;
  indices: Uint16Array;
  /** Number of vertices per ring. */
  ringLen: number;
  /** Number of total rings along Z. */
  totalRings: number;
}

/**
 * Generate a rounded-rectangle extrusion body matching the WebGL2 geometry.
 *
 * Cross-section profile is a rounded rectangle (wider in X). The profile is
 * extruded along Z from -0.5 to 0.5, with `nBody` interior rings between
 * the start and end rings. The vertex shader maps `position.z + 0.5` to
 * the parametric position `t` along the segment.
 */
export function generateBodyGeometry(
  hScale = 0.35,
  edgeSegments = 5,
  nBody = 3,
): BodyGeometry {
  const R = hScale / 2;       // corner radius
  const W = 0.5;              // half-width
  const cxL = -(W - R);
  const cxR =  W - R;

  // 2D cross-section perimeter (rounded rectangle, clockwise)
  const segs = edgeSegments;
  const ringLen = segs * 2 + 2;
  const profX = new Float32Array(ringLen);
  const profY = new Float32Array(ringLen);
  const normX = new Float32Array(ringLen);
  const normY = new Float32Array(ringLen);

  let pi = 0;
  // Top-right rounded corner (left-to-right top edge)
  for (let i = 0; i < segs; i++) {
    const t = -Math.PI / 2 + (i / (segs - 1)) * Math.PI;
    profX[pi] = cxR + R * Math.cos(t);
    profY[pi] = R * Math.sin(t);
    normX[pi] = Math.cos(t);
    normY[pi] = Math.sin(t);
    pi++;
  }
  // Top-left flat point
  profX[pi] = cxL; profY[pi] = R;
  normX[pi] = 0; normY[pi] = 1;
  pi++;
  // Left rounded corner (top-to-bottom)
  for (let i = 0; i < segs; i++) {
    const t = Math.PI / 2 + (i / (segs - 1)) * Math.PI;
    profX[pi] = cxL + R * Math.cos(t);
    profY[pi] = R * Math.sin(t);
    normX[pi] = Math.cos(t);
    normY[pi] = Math.sin(t);
    pi++;
  }
  // Bottom-right flat point (closing the loop)
  profX[pi] = cxR; profY[pi] = -R;
  normX[pi] = 0; normY[pi] = -1;
  pi++;

  const totalRings = nBody + 2;
  const verts = totalRings * ringLen;
  const interleaved = new Float32Array(verts * 6); // pos(3) + nrm(3)
  const indices: number[] = [];

  function setVert(ring: number, idx: number, x: number, y: number, z: number, nx: number, ny: number, nz: number) {
    const off = (ring * ringLen + idx) * 6;
    interleaved[off] = x;
    interleaved[off + 1] = y;
    interleaved[off + 2] = z;
    interleaved[off + 3] = nx;
    interleaved[off + 4] = ny;
    interleaved[off + 5] = nz;
  }

  // Generate rings
  for (let ring = 0; ring < totalRings; ring++) {
    const z = -0.5 + ring / (totalRings - 1);
    for (let i = 0; i < ringLen; i++) {
      setVert(ring, i, profX[i], profY[i], z, normX[i], normY[i], 0);
    }
  }

  // Connect adjacent rings with quads
  for (let r = 0; r < totalRings - 1; r++) {
    const r0 = r * ringLen;
    const r1 = (r + 1) * ringLen;
    for (let i = 0; i < ringLen; i++) {
      const nxt = (i + 1) % ringLen;
      indices.push(r0 + i, r1 + i, r1 + nxt);
      indices.push(r0 + i, r1 + nxt, r0 + nxt);
    }
  }

  return { interleaved, indices: new Uint16Array(indices), ringLen, totalRings };
}

export interface CapGeometry {
  /** Interleaved float32: [px, py, pz, nx, ny, nz] per vertex, 24 bytes/vertex. */
  interleaved: Float32Array;
  indices: Uint16Array;
}

/**
 * Generate a domed endcap geometry, matching the WebGL2 endcap.
 *
 * The dome sits on the XY plane at z=0 (rim) and bulges to z=1 (apex).
 * The vertex shader will position and orient it at segment endpoints.
 */
export function generateCapGeometry(
  hScale = 0.35,
  edgeSegments = 5,
  domeSegments = 4,
): CapGeometry {
  const R = hScale / 2;
  const W = 0.5;
  const cxL = -(W - R);
  const cxR =  W - R;
  const segs = edgeSegments;

  // Build 2D profile (same as body cross-section)
  const ringLen = segs * 2 + 2;
  const profX = new Float32Array(ringLen);
  const profY = new Float32Array(ringLen);
  const radNrmX = new Float32Array(ringLen);
  const radNrmY = new Float32Array(ringLen);

  let pi = 0;
  for (let i = 0; i < segs; i++) {
    const t = -Math.PI / 2 + (i / (segs - 1)) * Math.PI;
    profX[pi] = cxR + R * Math.cos(t);
    profY[pi] = R * Math.sin(t);
    radNrmX[pi] = Math.cos(t);
    radNrmY[pi] = Math.sin(t);
    pi++;
  }
  profX[pi] = cxL; profY[pi] = R; radNrmX[pi] = 0; radNrmY[pi] = 1; pi++;
  for (let i = 0; i < segs; i++) {
    const t = Math.PI / 2 + (i / (segs - 1)) * Math.PI;
    profX[pi] = cxL + R * Math.cos(t);
    profY[pi] = R * Math.sin(t);
    radNrmX[pi] = Math.cos(t);
    radNrmY[pi] = Math.sin(t);
    pi++;
  }
  profX[pi] = cxR; profY[pi] = -R; radNrmX[pi] = 0; radNrmY[pi] = -1; pi++;

  // Generate dome: rings at z = 0..1 with scale = sqrt(1 - z*z)
  const domeRings = domeSegments;
  const verts = domeRings * ringLen + 1; // +1 for apex
  const interleaved = new Float32Array(verts * 6);
  const indices: number[] = [];

  function setVert(idx: number, x: number, y: number, z: number, nx: number, ny: number, nz: number) {
    const off = idx * 6;
    interleaved[off] = x; interleaved[off + 1] = y; interleaved[off + 2] = z;
    interleaved[off + 3] = nx; interleaved[off + 4] = ny; interleaved[off + 5] = nz;
  }

  for (let j = 0; j < domeRings; j++) {
    const z = j / domeRings;
    const scale = Math.sqrt(Math.max(0, 1 - z * z));
    const w = 1.0 - z; // blend weight: 1 at rim, 0 at apex
    for (let i = 0; i < ringLen; i++) {
      const nx = radNrmX[i] * w;
      const ny = radNrmY[i] * w;
      const nz2 = z;
      const nl = Math.sqrt(nx * nx + ny * ny + nz2 * nz2);
      setVert(j * ringLen + i, profX[i] * scale, profY[i] * scale, z,
        nx / nl, ny / nl, nz2 / nl);
    }
  }

  // Apex
  const apexIdx = domeRings * ringLen;
  setVert(apexIdx, 0, 0, 1, 0, 0, 1);

  // Connect adjacent rings
  for (let j = 0; j < domeRings - 1; j++) {
    const r0 = j * ringLen;
    const r1 = (j + 1) * ringLen;
    for (let i = 0; i < ringLen; i++) {
      const nxt = (i + 1) % ringLen;
      indices.push(r0 + i, r1 + i, r1 + nxt);
      indices.push(r0 + i, r1 + nxt, r0 + nxt);
    }
  }

  // Connect last ring to apex
  const lastRing = (domeRings - 1) * ringLen;
  for (let i = 0; i < ringLen; i++) {
    const nxt = (i + 1) % ringLen;
    indices.push(lastRing + i, lastRing + nxt, apexIdx);
  }

  return { interleaved, indices: new Uint16Array(indices) };
}
