// geometry.ts

// ── Shape constants ──
const PROFILE_FLAT_HALF_WIDTH = 0.35;   // half‑width of the flat top/bottom
const DEFAULT_HSCALE = 0.35;
const DEFAULT_EDGE_SEGMENTS = 5;
const DEFAULT_BULGE_ANGLE_DEG = 40;
const DEFAULT_BULGE_RATIO = 0.5;

// ── Profile helper ──
interface Profile {
  /** Interleaved [x, y] for each vertex. */
  positions: Float32Array;   // length = ringLen * 2
  /** Normalised outward normals [nx, ny]. */
  normals: Float32Array;     // length = ringLen * 2
  ringLen: number;
}

/**
 * Build the 2D cross‑section of the flattened capsule.
 *
 * The shape has:
 *  - a flat top    (y = +halfHeight, x from -flatHalfW to +flatHalfW)
 *  - a flat bottom (y = -halfHeight, same x range)
 *  - side curves that bulge outward, starting with a 40° tangent at the
 *    sharp corner where the flat ends (custom Bézier).
 *
 * The profile is traversed **clockwise** starting at the top‑right corner.
 * Outward normals are the left‑hand perpendicular to the curve tangent
 * (interior is to the right of the direction of travel).
 */
function generateProfile(
  flatHalfW: number,
  halfHeight: number,
  maxHalfW: number,
  bulgeRad: number,
  edgeSegments: number,
): Profile {
  const segs = edgeSegments;
  const ringLen = 4 * segs + 4;   // 4 corners + 4 curves × segs

  const posX = new Float32Array(ringLen);
  const posY = new Float32Array(ringLen);
  const normX = new Float32Array(ringLen);
  const normY = new Float32Array(ringLen);

  // ---- Bézier helpers ----
  const bezier = (
    p0: [number, number], p1: [number, number],
    p2: [number, number], p3: [number, number], t: number,
  ): [number, number] => {
    const mt = 1 - t;
    const mt2 = mt * mt, mt3 = mt2 * mt;
    const t2 = t * t,  t3  = t2 * t;
    return [
      mt3 * p0[0] + 3 * mt2 * t * p1[0] + 3 * mt * t2 * p2[0] + t3 * p3[0],
      mt3 * p0[1] + 3 * mt2 * t * p1[1] + 3 * mt * t2 * p2[1] + t3 * p3[1],
    ];
  };

  const bezierDeriv = (
    p0: [number, number], p1: [number, number],
    p2: [number, number], p3: [number, number], t: number,
  ): [number, number] => {
    const mt = 1 - t;
    return [
      3 * mt * mt * (p1[0] - p0[0]) + 6 * mt * t * (p2[0] - p1[0]) + 3 * t * t * (p3[0] - p2[0]),
      3 * mt * mt * (p1[1] - p0[1]) + 6 * mt * t * (p2[1] - p1[1]) + 3 * t * t * (p3[1] - p2[1]),
    ];
  };

  // ---- Control points for the right‑side curves ----
  const ctrlDist = (maxHalfW - flatHalfW) * 0.8;
  const cosA = Math.cos(bulgeRad), sinA = Math.sin(bulgeRad);

  // top‑right curve: (W, H) → (maxW, 0)
  const p0: [number, number] = [flatHalfW, halfHeight];
  const p1: [number, number] = [flatHalfW + ctrlDist * cosA, halfHeight - ctrlDist * sinA];
  const p2: [number, number] = [maxHalfW, halfHeight * 0.5];
  const p3: [number, number] = [maxHalfW, 0];

  // bottom‑right curve: (maxW, 0) → (W, -H)
  const bp0: [number, number] = [maxHalfW, 0];
  const bp1: [number, number] = [maxHalfW, -halfHeight * 0.5];
  const bp2: [number, number] = [flatHalfW + ctrlDist * cosA, -halfHeight + ctrlDist * sinA];
  const bp3: [number, number] = [flatHalfW, -halfHeight];

  // ---- Add a point with its outward normal ----
  let i = 0;
  const addPt = (x: number, y: number, dx: number, dy: number) => {
    posX[i] = x; posY[i] = y;
    // Outward normal = left‑hand perpendicular to (dx, dy)
    const len = Math.hypot(dx, dy);
    if (len > 1e-8) {
      normX[i] = -dy / len;
      normY[i] =  dx / len;
    } else {
      // Fallback for degenerate edge – points away from centre
      normX[i] = 0;
      normY[i] = (y >= 0 ? 1 : -1);
    }
    i++;
  };

  // ---- Generate perimeter clockwise ----
  // 1. Top flat, right edge (moving right)
  addPt( flatHalfW,  halfHeight,  1, 0);
  // 2. Top‑right curve
  for (let s = 1; s <= segs; s++) {
    const t = s / segs;
    const [x, y] = bezier(p0, p1, p2, p3, t);
    const [dx, dy] = bezierDeriv(p0, p1, p2, p3, t);
    addPt(x, y, dx, dy);
  }
  // 3. Bottom‑right curve
  for (let s = 1; s <= segs; s++) {
    const t = s / segs;
    const [x, y] = bezier(bp0, bp1, bp2, bp3, t);
    const [dx, dy] = bezierDeriv(bp0, bp1, bp2, bp3, t);
    addPt(x, y, dx, dy);
  }
  // 4. Bottom flat, right‑to‑left
  addPt( flatHalfW, -halfHeight, -1, 0);
  // 5. Bottom flat, left half (continues moving left)
  addPt(-flatHalfW, -halfHeight, -1, 0);
  // 6. Bottom‑left curve: from (-flatHalfW, -halfHeight) up to (-maxHalfW, 0)
  //    Reverse the mirrored bottom‑right control points.
  for (let s = 1; s <= segs; s++) {
    const t = s / segs;
    const [x, y] = bezier(
      [-bp3[0], bp3[1]],   // start = mirror of bp3
      [-bp2[0], bp2[1]],   // control 1 = mirror of bp2
      [-bp1[0], bp1[1]],   // control 2 = mirror of bp1
      [-bp0[0], bp0[1]],   // end   = mirror of bp0
      t,
    );
    const [dx, dy] = bezierDeriv(
      [-bp3[0], bp3[1]],
      [-bp2[0], bp2[1]],
      [-bp1[0], bp1[1]],
      [-bp0[0], bp0[1]],
      t,
    );
    addPt(x, y, dx, dy);
  }

  // 7. Top‑left curve: from (-maxHalfW, 0) up to (-flatHalfW, halfHeight)
  //    Reverse the mirrored top‑right control points.
  for (let s = 1; s <= segs; s++) {
    const t = s / segs;
    const [x, y] = bezier(
      [-p3[0], p3[1]],   // start = mirror of p3
      [-p2[0], p2[1]],
      [-p1[0], p1[1]],
      [-p0[0], p0[1]],   // end   = mirror of p0
      t,
    );
    const [dx, dy] = bezierDeriv(
      [-p3[0], p3[1]],
      [-p2[0], p2[1]],
      [-p1[0], p1[1]],
      [-p0[0], p0[1]],
      t,
    );
    addPt(x, y, dx, dy);
  }
  // 8. Top flat, left edge (closes the loop)
  addPt(-flatHalfW, halfHeight, 1, 0);

  // Pack positions and normals into interleaved‑pair arrays
  const posOut = new Float32Array(ringLen * 2);
  const nrmOut = new Float32Array(ringLen * 2);
  for (let j = 0; j < ringLen; j++) {
    posOut[j * 2] = posX[j]; posOut[j * 2 + 1] = posY[j];
    nrmOut[j * 2] = normX[j]; nrmOut[j * 2 + 1] = normY[j];
  }

  return { positions: posOut, normals: nrmOut, ringLen };
}

// ──────────────────────────────────────────────
//  Body geometry (extruded “tube”)
// ──────────────────────────────────────────────

export interface BodyGeometry {
  /** Interleaved float32: [px, py, pz, nx, ny, nz] per vertex, 24 bytes/vertex. */
  interleaved: Float32Array;
  indices: Uint16Array;
  ringLen: number;
  totalRings: number;
}

/**
 * Generate a flattened‑capsule extrusion body.
 *
 * @param flatHalfW  Half‑width of the flat top/bottom.
 * @param hScale     Overall height; halfHeight = hScale / 2.
 * @param edgeSegments  Number of vertices along each quadrant curve.
 * @param nBody      Number of interior rings (0 = only end rings).
 * @param bulgeAngleDeg  Angle from horizontal where the curve starts.
 * @param bulgeRatio     Maximum extra half‑width bulge factor.
 */
export function generateBodyGeometry(
  flatHalfW = PROFILE_FLAT_HALF_WIDTH,
  hScale = DEFAULT_HSCALE,
  edgeSegments = DEFAULT_EDGE_SEGMENTS,
  nBody = 3,
  bulgeAngleDeg = DEFAULT_BULGE_ANGLE_DEG,
  bulgeRatio = DEFAULT_BULGE_RATIO,
): BodyGeometry {
  const halfHeight = hScale / 2;
  const maxHalfW = flatHalfW * (1 + bulgeRatio);
  const bulgeRad = bulgeAngleDeg * Math.PI / 180;

  const profile = generateProfile(flatHalfW, halfHeight, maxHalfW, bulgeRad, edgeSegments);
  const { ringLen, positions: profPos, normals: profNrm } = profile;

  const totalRings = nBody + 2;   // start ring, end ring, plus nBody interior
  const verts = totalRings * ringLen;
  const interleaved = new Float32Array(verts * 6); // 3 pos + 3 nrm
  const indices: number[] = [];

  const setVert = (ring: number, idx: number, x: number, y: number, z: number,
                   nx: number, ny: number, nz: number) => {
    const off = (ring * ringLen + idx) * 6;
    interleaved[off + 0] = x;
    interleaved[off + 1] = y;
    interleaved[off + 2] = z;
    interleaved[off + 3] = nx;
    interleaved[off + 4] = ny;
    interleaved[off + 5] = nz;
  };

  // Place rings along Z from -0.5 to +0.5
  for (let ring = 0; ring < totalRings; ring++) {
    const z = -0.5 + ring / (totalRings - 1);
    for (let i = 0; i < ringLen; i++) {
      const x = profPos[i * 2];
      const y = profPos[i * 2 + 1];
      const nx = profNrm[i * 2];
      const ny = profNrm[i * 2 + 1];
      setVert(ring, i, x, y, z, nx, ny, 0);
    }
  }

  // Quad indices connecting adjacent rings
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

// ──────────────────────────────────────────────
//  End‑cap geometry (domed caps)
// ──────────────────────────────────────────────

export interface CapGeometry {
  /** Interleaved float32: [px, py, pz, nx, ny, nz] per vertex, 24 bytes/vertex. */
  interleaved: Float32Array;
  indices: Uint16Array;
}

/**
 * Generate a domed end‑cap.
 *
 * The dome bulges from z=0 (rim) to z=1 (apex).  It uses the same cross‑section
 * profile as the body and scales it by sqrt(1 - z²) to create a smooth,
 * hemispherical‑like surface.  Normals are pre‑computed as a blend between the
 * 2D profile normal and the dome’s vertical direction.
 */
export function generateCapGeometry(
  flatHalfW = PROFILE_FLAT_HALF_WIDTH,
  hScale = DEFAULT_HSCALE,
  edgeSegments = DEFAULT_EDGE_SEGMENTS,
  domeSegments = 4,
  bulgeAngleDeg = DEFAULT_BULGE_ANGLE_DEG,
  bulgeRatio = DEFAULT_BULGE_RATIO,
): CapGeometry {
  const halfHeight = hScale / 2;
  const maxHalfW = flatHalfW * (1 + bulgeRatio);
  const bulgeRad = bulgeAngleDeg * Math.PI / 180;

  const profile = generateProfile(flatHalfW, halfHeight, maxHalfW, bulgeRad, edgeSegments);
  const { ringLen, positions: profPos, normals: profNrm } = profile;

  const domeRings = domeSegments;
  const verts = domeRings * ringLen + 1;   // +1 apex
  const interleaved = new Float32Array(verts * 6);
  const indices: number[] = [];

  const setVert = (idx: number, x: number, y: number, z: number,
                   nx: number, ny: number, nz: number) => {
    const off = idx * 6;
    interleaved[off + 0] = x;
    interleaved[off + 1] = y;
    interleaved[off + 2] = z;
    interleaved[off + 3] = nx;
    interleaved[off + 4] = ny;
    interleaved[off + 5] = nz;
  };

  // Generate dome rings
  for (let j = 0; j < domeRings; j++) {
    const z = j / domeRings;
    const scale = Math.sqrt(Math.max(0, 1 - z * z)); // hemispherical scaling
    const w = 1.0 - z; // blend weight: 1 at rim, 0 at apex

    for (let i = 0; i < ringLen; i++) {
      const x = profPos[i * 2] * scale;
      const y = profPos[i * 2 + 1] * scale;
      // Blend between the 2D profile normal and the vertical (0,0,1)
      let nx = profNrm[i * 2]     * w;
      let ny = profNrm[i * 2 + 1] * w;
      let nz = z;
      const nl = Math.sqrt(nx * nx + ny * ny + nz * nz);
      setVert(j * ringLen + i, x, y, z, nx / nl, ny / nl, nz / nl);
    }
  }

  // Apex
  const apexIdx = domeRings * ringLen;
  setVert(apexIdx, 0, 0, 1, 0, 0, 1);

  // Indices – rings (flipped winding: body uses r0,r1,r1_next and r0,r1_next,r0_next
  // but the dome's curvature makes the opposite winding face outward)
  for (let j = 0; j < domeRings - 1; j++) {
    const r0 = j * ringLen;
    const r1 = (j + 1) * ringLen;
    for (let i = 0; i < ringLen; i++) {
      const nxt = (i + 1) % ringLen;
      indices.push(r0 + i, r1 + nxt, r1 + i);
      indices.push(r0 + i, r0 + nxt, r1 + nxt);
    }
  }

  // Indices – last ring to apex (also flipped)
  const lastRing = (domeRings - 1) * ringLen;
  for (let i = 0; i < ringLen; i++) {
    const nxt = (i + 1) % ringLen;
    indices.push(lastRing + i, apexIdx, lastRing + nxt);
  }

  return { interleaved, indices: new Uint16Array(indices) };
}

// ──────────────────────────────────────────────
//  LOD helpers
// ──────────────────────────────────────────────

/** All 3 LOD body geometries (matching WebGL2 LOD_BODY_GEO). */
export function generateAllBodyGeometries(): [BodyGeometry, BodyGeometry, BodyGeometry] {
  return [
    generateBodyGeometry(PROFILE_FLAT_HALF_WIDTH, DEFAULT_HSCALE, 5, 3),  // LOD 0
    generateBodyGeometry(PROFILE_FLAT_HALF_WIDTH, DEFAULT_HSCALE, 3, 1),  // LOD 1
    generateBodyGeometry(PROFILE_FLAT_HALF_WIDTH, DEFAULT_HSCALE, 5, 0),  // LOD 2
  ];
}

/** LOD 0 and 1 cap geometries (LOD 2 uses no caps). */
export function generateAllCapGeometries(): [CapGeometry, CapGeometry] {
  return [
    generateCapGeometry(PROFILE_FLAT_HALF_WIDTH, DEFAULT_HSCALE, 5, 4),  // LOD 0
    generateCapGeometry(PROFILE_FLAT_HALF_WIDTH, DEFAULT_HSCALE, 3, 2),  // LOD 1
  ];
}
