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
