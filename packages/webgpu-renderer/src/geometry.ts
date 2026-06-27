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
 * Generate a flattened-capsule extrusion body.
 *
 * Cross-section profile: flat top and bottom with elliptical sides
 * that transition from the flat at ~40° and bulge outward (ellipse-like).
 * The shape looks like a pill / capsule that's been flattened.
 *
 * Parameters:
 *   hScale — controls the overall height (and indirectly the bulge angle)
 *   edgeSegments — number of vertices along each quadrant of the curve
 *   nBody — number of interior rings along the extrusion
 *   bulgeAngleDeg — angle from horizontal at the flat-to-curve transition
 *   bulgeRatio — how far the side bulges out (fraction of half-width)
 */
export function generateBodyGeometry(
  hScale = 0.35,
  edgeSegments = 5,
  nBody = 3,
  bulgeAngleDeg = 40,
  bulgeRatio = 0.5,
): BodyGeometry {
  const halfHeight = hScale / 2;
  const flatHalfW = 0.325;       // half-width of the flat top/bottom (matches old R profile)
  const maxHalfW = flatHalfW * (1 + bulgeRatio); // max half-width at waist

  const bulgeRad = bulgeAngleDeg * Math.PI / 180;

  // Generate right-half profile points: from top-center to bottom-center
  // Top flat (center → edge): y = halfHeight, x from 0 to flatHalfW
  // Top-right curve: Bezier from (flatHalfW, halfHeight) to (maxHalfW, 0)
  // Bottom-right curve: Bezier from (maxHalfW, 0) to (flatHalfW, -halfHeight)
  // Bottom flat: (flatHalfW → 0, -halfHeight)

  const segs = edgeSegments; // vertices per quadrant
  // Total ring = top flat (1) + top-right curve (segs) + bottom-right curve (segs) + bottom flat (1) = 2*segs + 2
  // Wait, we need to go around the whole perimeter:
  // top flat right (from center to edge) + top-right curve + bottom-right curve + bottom flat right + bottom flat left + bottom-left curve + top-left curve + top flat left
  // = 1 + segs + segs + 1 + 1 + segs + segs + 1 = 4*segs + 4 = 4*(segs+1)  
  // Actually simpler: ring = flat top (W, H) → right curve → flat bottom (W, -H) → left side back

  // Approach: generate the perimeter clockwise starting at top-left corner of flat
  // Top flat: left-to-right
  // Right curve: top-right to bottom-right (quarter Bezier × 2)
  // Bottom flat: right-to-left
  // Left curve: bottom-left to top-left (quarter Bezier × 2)

  // Evaluate cubic Bezier at parameter t ∈ [0,1]
  function bezier(p0: number[], p1: number[], p2: number[], p3: number[], t: number): [number, number] {
    const mt = 1 - t;
    const mt2 = mt * mt, mt3 = mt2 * mt;
    const t2 = t * t, t3 = t2 * t;
    return [
      mt3 * p0[0] + 3 * mt2 * t * p1[0] + 3 * mt * t2 * p2[0] + t3 * p3[0],
      mt3 * p0[1] + 3 * mt2 * t * p1[1] + 3 * mt * t2 * p2[1] + t3 * p3[1],
    ];
  }

  // Derivative of cubic Bezier at parameter t
  function bezierDeriv(p0: number[], p1: number[], p2: number[], p3: number[], t: number): [number, number] {
    const mt = 1 - t;
    return [
      3 * mt * mt * (p1[0] - p0[0]) + 6 * mt * t * (p2[0] - p1[0]) + 3 * t * t * (p3[0] - p2[0]),
      3 * mt * mt * (p1[1] - p0[1]) + 6 * mt * t * (p2[1] - p1[1]) + 3 * t * t * (p3[1] - p2[1]),
    ];
  }

  // Bezier control points for the top-right quadrant (flat edge → widest point)
  // Tangent at start: 40° from horizontal (dx>0, dy<0 = going down-right)
  // Tangent at end: vertical (pointing down)
  const ctrlDist = (maxHalfW - flatHalfW) * 0.8;
  const cosA = Math.cos(bulgeRad), sinA = Math.sin(bulgeRad);
  const p0: [number, number] = [flatHalfW, halfHeight];
  const p1: [number, number] = [flatHalfW + ctrlDist * cosA, halfHeight - ctrlDist * sinA];
  const p2: [number, number] = [maxHalfW, halfHeight * 0.5];
  const p3: [number, number] = [maxHalfW, 0];

  // Bottom-right quadrant: (maxW, 0) → (W, -H)
  const bp0: [number, number] = [maxHalfW, 0];
  const bp1: [number, number] = [maxHalfW, -halfHeight * 0.5];
  const bp2: [number, number] = [flatHalfW + ctrlDist * cosA, -halfHeight + ctrlDist * sinA];
  const bp3: [number, number] = [flatHalfW, -halfHeight];

  // Compute ring vertex count
  const maxRingLen = 4 * segs + 4;
  const profX = new Float32Array(maxRingLen);
  const profY = new Float32Array(maxRingLen);
  const normX = new Float32Array(maxRingLen);
  const normY = new Float32Array(maxRingLen);

  let pi = 0;

  // Helper: add profile point with computed outward normal
  function addPt(x: number, y: number, dx: number, dy: number) {
    const len = Math.sqrt(dx * dx + dy * dy);
    const nx = len > 1e-8 ? -dy / len : 0;
    const ny = len > 1e-8 ? dx / len : (y >= 0 ? 1 : -1);
    profX[pi] = x; profY[pi] = y;
    normX[pi] = nx; normY[pi] = ny;
    pi++;
  }

  // Generate perimeter clockwise, starting at top-right corner:
  // right flat edge → top-right curve → bottom-right curve → bottom edge
  // → bottom-left → bottom-left curve → top-left curve → back to top edge

  // Top-right edge (included as start and end of ring closure via modulo)
  addPt(flatHalfW, halfHeight, 1, 0);

  // Top-right curve: (W, H) → (maxW, 0)
  for (let i = 1; i <= segs; i++) {
    const t = i / segs;
    const [x, y] = bezier(p0, p1, p2, p3, t);
    const [dx, dy] = bezierDeriv(p0, p1, p2, p3, t);
    addPt(x, y, dx, dy);
  }

  // Bottom-right curve: (maxW, 0) → (W, -H)
  for (let i = 1; i <= segs; i++) {
    const t = i / segs;
    const [x, y] = bezier(bp0, bp1, bp2, bp3, t);
    const [dx, dy] = bezierDeriv(bp0, bp1, bp2, bp3, t);
    addPt(x, y, dx, dy);
  }

  // Bottom-right edge
  addPt(flatHalfW, -halfHeight, -1, 0);

  // Bottom-left edge
  addPt(-flatHalfW, -halfHeight, -1, 0);

  // Bottom-left curve: (-W, -H) → (-maxW, 0)
  for (let i = 1; i <= segs; i++) {
    const t = i / segs;
    const [x, y] = bezier([-bp0[0], bp0[1]], [-bp1[0], bp1[1]], [-bp2[0], bp2[1]], [-bp3[0], bp3[1]], t);
    const [dx, dy] = bezierDeriv([-bp0[0], bp0[1]], [-bp1[0], bp1[1]], [-bp2[0], bp2[1]], [-bp3[0], bp3[1]], t);
    addPt(x, y, dx, dy);
  }

  // Top-left curve: (-maxW, 0) → (-W, H)
  for (let i = 1; i <= segs; i++) {
    const t = i / segs;
    const [x, y] = bezier([-p0[0], p0[1]], [-p1[0], p1[1]], [-p2[0], p2[1]], [-p3[0], p3[1]], t);
    const [dx, dy] = bezierDeriv([-p0[0], p0[1]], [-p1[0], p1[1]], [-p2[0], p2[1]], [-p3[0], p3[1]], t);
    addPt(x, y, dx, dy);
  }

  // Top-left edge (closes the loop via modulo indexing)
  addPt(-flatHalfW, halfHeight, 1, 0);

  const ringLen = pi;

  // Close the loop — last point should connect to first (top-center)
  // But top-center is at index 0, so we're already looped

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
  bulgeAngleDeg = 40,
  bulgeRatio = 0.5,
): CapGeometry {
  const halfHeight = hScale / 2;
  const flatHalfW = 0.35;
  const maxHalfW = flatHalfW * (1 + bulgeRatio);
  const bulgeRad = bulgeAngleDeg * Math.PI / 180;
  const segs = edgeSegments;

  // Same Bezier-based profile as body geometry
  function bezier(p0: number[], p1: number[], p2: number[], p3: number[], t: number): [number, number] {
    const mt = 1 - t, mt2 = mt * mt, mt3 = mt2 * mt;
    const t2 = t * t, t3 = t2 * t;
    return [
      mt3 * p0[0] + 3 * mt2 * t * p1[0] + 3 * mt * t2 * p2[0] + t3 * p3[0],
      mt3 * p0[1] + 3 * mt2 * t * p1[1] + 3 * mt * t2 * p2[1] + t3 * p3[1],
    ];
  }

  function bezierDeriv(p0: number[], p1: number[], p2: number[], p3: number[], t: number): [number, number] {
    const mt = 1 - t;
    return [
      3 * mt * mt * (p1[0] - p0[0]) + 6 * mt * t * (p2[0] - p1[0]) + 3 * t * t * (p3[0] - p2[0]),
      3 * mt * mt * (p1[1] - p0[1]) + 6 * mt * t * (p2[1] - p1[1]) + 3 * t * t * (p3[1] - p2[1]),
    ];
  }

  const ctrlDist = (maxHalfW - flatHalfW) * 0.8;
  const cosA = Math.cos(bulgeRad), sinA = Math.sin(bulgeRad);
  const p0: [number, number] = [flatHalfW, halfHeight];
  const p1: [number, number] = [flatHalfW + ctrlDist * cosA, halfHeight - ctrlDist * sinA];
  const p2: [number, number] = [maxHalfW, halfHeight * 0.5];
  const p3: [number, number] = [maxHalfW, 0];
  const bp0: [number, number] = [maxHalfW, 0];
  const bp1: [number, number] = [maxHalfW, -halfHeight * 0.5];
  const bp2: [number, number] = [flatHalfW + ctrlDist * cosA, -halfHeight + ctrlDist * sinA];
  const bp3: [number, number] = [flatHalfW, -halfHeight];

  // Same clockwise perimeter as body geometry (simplified — no center vertices)
  const capRingLen = 4 * segs + 4; // top edge + 4 curves + bottom edges + closing
  const profX = new Float32Array(capRingLen);
  const profY = new Float32Array(capRingLen);
  const radNrmX = new Float32Array(capRingLen);
  const radNrmY = new Float32Array(capRingLen);

  let pi = 0;
  function addCPt(x: number, y: number, dx: number, dy: number) {
    const len = Math.sqrt(dx * dx + dy * dy);
    const nx = len > 1e-8 ? -dy / len : 0;
    const ny = len > 1e-8 ? dx / len : (y >= 0 ? 1 : -1);
    profX[pi] = x; profY[pi] = y; radNrmX[pi] = nx; radNrmY[pi] = ny; pi++;
  }

  addCPt(flatHalfW, halfHeight, 1, 0);
  for (let i = 1; i <= segs; i++) {
    const t = i / segs;
    const [x, y] = bezier(p0, p1, p2, p3, t);
    const [dx, dy] = bezierDeriv(p0, p1, p2, p3, t);
    addCPt(x, y, dx, dy);
  }
  for (let i = 1; i <= segs; i++) {
    const t = i / segs;
    const [x, y] = bezier(bp0, bp1, bp2, bp3, t);
    const [dx, dy] = bezierDeriv(bp0, bp1, bp2, bp3, t);
    addCPt(x, y, dx, dy);
  }
  addCPt(flatHalfW, -halfHeight, -1, 0);
  addCPt(-flatHalfW, -halfHeight, -1, 0);
  for (let i = 1; i <= segs; i++) {
    const t = i / segs;
    const [x, y] = bezier([-bp0[0], bp0[1]], [-bp1[0], bp1[1]], [-bp2[0], bp2[1]], [-bp3[0], bp3[1]], t);
    const [dx, dy] = bezierDeriv([-bp0[0], bp0[1]], [-bp1[0], bp1[1]], [-bp2[0], bp2[1]], [-bp3[0], bp3[1]], t);
    addCPt(x, y, dx, dy);
  }
  for (let i = 1; i <= segs; i++) {
    const t = i / segs;
    const [x, y] = bezier([-p0[0], p0[1]], [-p1[0], p1[1]], [-p2[0], p2[1]], [-p3[0], p3[1]], t);
    const [dx, dy] = bezierDeriv([-p0[0], p0[1]], [-p1[0], p1[1]], [-p2[0], p2[1]], [-p3[0], p3[1]], t);
    addCPt(x, y, dx, dy);
  }
  // Top-left edge (closes the ring via modulo)
  addCPt(-flatHalfW, halfHeight, 1, 0);

  const capRL = pi;
  console.assert(capRL === capRingLen, `cap ringLen mismatch: ${capRL} vs ${capRingLen}`);

  // Generate dome: rings at z = 0..1 with scale = sqrt(1 - z*z)
  const domeRings = domeSegments;
  const verts = domeRings * capRL + 1; // +1 for apex
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
    for (let i = 0; i < capRL; i++) {
      const nx = radNrmX[i] * w;
      const ny = radNrmY[i] * w;
      const nz2 = z;
      const nl = Math.sqrt(nx * nx + ny * ny + nz2 * nz2);
      setVert(j * capRL + i, profX[i] * scale, profY[i] * scale, z,
        nx / nl, ny / nl, nz2 / nl);
    }
  }

  // Apex
  const apexIdx = domeRings * capRL;
  setVert(apexIdx, 0, 0, 1, 0, 0, 1);

  // Connect adjacent rings
  for (let j = 0; j < domeRings - 1; j++) {
    const r0 = j * capRL;
    const r1 = (j + 1) * capRL;
    for (let i = 0; i < capRL; i++) {
      const nxt = (i + 1) % capRL;
      indices.push(r0 + i, r1 + i, r1 + nxt);
      indices.push(r0 + i, r1 + nxt, r0 + nxt);
    }
  }

  // Connect last ring to apex
  const lastRing = (domeRings - 1) * capRL;
  for (let i = 0; i < capRL; i++) {
    const nxt = (i + 1) % capRL;
    indices.push(lastRing + i, lastRing + nxt, apexIdx);
  }

  return { interleaved, indices: new Uint16Array(indices) };
}

/** All 3 LOD body geometries (matching WebGL2 LOD_BODY_GEO). */
export function generateAllBodyGeometries(): [BodyGeometry, BodyGeometry, BodyGeometry] {
  return [
    generateBodyGeometry(0.35, 5, 3),  // LOD 0: full detail
    generateBodyGeometry(0.35, 3, 1),  // LOD 1: reduced
    generateBodyGeometry(0.35, 5, 0),  // LOD 2: full cross-section, no interior rings
  ];
}

/** LOD 0 and 1 cap geometries (LOD 2 has no caps). */
export function generateAllCapGeometries(): [CapGeometry, CapGeometry] {
  return [
    generateCapGeometry(0.35, 5, 4),  // LOD 0: full
    generateCapGeometry(0.35, 3, 2),  // LOD 1: reduced
  ];
}
