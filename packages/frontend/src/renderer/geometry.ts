export interface GeometryData {
  positions: Float32Array;
  normals: Float32Array;
  indices: Uint16Array;
}

export function defaultCrossSectionGeometry(hScale = 0.35, edgeSegments = 5): GeometryData {
  const R = hScale / 2;
  const W = 0.5;
  const cxL = -(W - R);
  const cxR =  W - R;

  const segs = edgeSegments;
  const ringLen = segs * 2 + 2;
  const pos: number[] = [];
  const nrm: number[] = [];

  for (let i = 0; i < segs; i++) {
    const t = -Math.PI / 2 + (i / (segs - 1)) * Math.PI;
    pos.push(cxR + R * Math.cos(t), R * Math.sin(t));
    nrm.push(Math.cos(t), Math.sin(t));
  }
  pos.push(cxL, R);  nrm.push(0, 1);
  for (let i = 0; i < segs; i++) {
    const t = Math.PI / 2 + (i / (segs - 1)) * Math.PI;
    pos.push(cxL + R * Math.cos(t), R * Math.sin(t));
    nrm.push(Math.cos(t), Math.sin(t));
  }
  pos.push(cxR, -R);  nrm.push(0, -1);

  const posArr: number[] = [];
  const nrmArr: number[] = [];
  const idx: number[] = [];
  const nBody = 3;

  const e0 = 0;
  for (let i = 0; i < ringLen; i++) {
    posArr.push(pos[i * 2], pos[i * 2 + 1], -0.5);
    nrmArr.push(nrm[i * 2], nrm[i * 2 + 1], 0);
  }
  const e1 = ringLen;
  for (let i = 0; i < ringLen; i++) {
    posArr.push(pos[i * 2], pos[i * 2 + 1], 0.5);
    nrmArr.push(nrm[i * 2], nrm[i * 2 + 1], 0);
  }
  const body0 = 2 * ringLen;
  for (let j = 0; j < nBody; j++) {
    const z = -0.5 + (j + 1) / (nBody + 1);
    for (let i = 0; i < ringLen; i++) {
      posArr.push(pos[i * 2], pos[i * 2 + 1], z);
      nrmArr.push(nrm[i * 2], nrm[i * 2 + 1], 0);
    }
  }

  function cr(r0: number, r1: number) {
    for (let i = 0; i < ringLen; i++) {
      const n = (i + 1) % ringLen;
      idx.push(r0 + i, r1 + i, r1 + n);
      idx.push(r0 + i, r1 + n, r0 + n);
    }
  }
  cr(e0, body0);
  for (let j = 0; j < nBody - 1; j++)
    cr(body0 + j * ringLen, body0 + (j + 1) * ringLen);
  cr(body0 + (nBody - 1) * ringLen, e1);

  const triCount = idx.length / 3;
  console.log(`[geometry] body: ${triCount} tris, ${ringLen} verts/ring, ${nBody + 2} rings`);
  return { positions: new Float32Array(posArr), normals: new Float32Array(nrmArr), indices: new Uint16Array(idx) };
}

export function defaultEndcapGeometry(hScale = 0.35, edgeSegments = 5, domeSegments = 2): GeometryData {
  const R = hScale / 2;
  const W = 0.5;
  const cxL = -(W - R);
  const cxR = W - R;
  const segs = edgeSegments;
  const ringLen = segs * 2 + 2;

  const profX: number[] = [];
  const profY: number[] = [];
  for (let i = 0; i < segs; i++) {
    const t = -Math.PI / 2 + (i / (segs - 1)) * Math.PI;
    profX.push(cxR + R * Math.cos(t));
    profY.push(R * Math.sin(t));
  }
  profX.push(cxL); profY.push(R);
  for (let i = 0; i < segs; i++) {
    const t = Math.PI / 2 + (i / (segs - 1)) * Math.PI;
    profX.push(cxL + R * Math.cos(t));
    profY.push(R * Math.sin(t));
  }
  profX.push(cxR); profY.push(-R);

  // Precompute radial normals for each profile point
  const radNrm: number[] = [];
  for (let i = 0; i < segs; i++) {
    const t = -Math.PI / 2 + (i / (segs - 1)) * Math.PI;
    radNrm.push(Math.cos(t), Math.sin(t));
  }
  radNrm.push(0, 1);
  for (let i = 0; i < segs; i++) {
    const t = Math.PI / 2 + (i / (segs - 1)) * Math.PI;
    radNrm.push(Math.cos(t), Math.sin(t));
  }
  radNrm.push(0, -1);

  const pos: number[] = [];
  const nrm: number[] = [];
  const idx: number[] = [];

  for (let j = 0; j < domeSegments; j++) {
    const z = j / domeSegments;
    const scale = Math.sqrt(Math.max(0, 1 - z * z));
    const w = 1.0 - z; // blend weight: 1 at rim, 0 at apex
    for (let i = 0; i < ringLen; i++) {
      pos.push(profX[i] * scale, profY[i] * scale, z);
      // Normal: blend from purely radial at rim to (0,0,1) at apex
      const nx = radNrm[i * 2] * w;
      const ny = radNrm[i * 2 + 1] * w;
      const nz = z;
      const nl = Math.sqrt(nx * nx + ny * ny + nz * nz);
      nrm.push(nx / nl, ny / nl, nz / nl);
    }
  }

  const apex = domeSegments * ringLen;
  pos.push(0, 0, 1);
  nrm.push(0, 0, 1); // apex normal

  for (let j = 0; j < domeSegments - 1; j++) {
    const r0 = j * ringLen;
    const r1 = (j + 1) * ringLen;
    for (let i = 0; i < ringLen; i++) {
      const n = (i + 1) % ringLen;
      idx.push(r0 + i, r1 + n, r1 + i);
      idx.push(r0 + i, r0 + n, r1 + n);
    }
  }

  const lastRing = (domeSegments - 1) * ringLen;
  for (let i = 0; i < ringLen; i++) {
    const n = (i + 1) % ringLen;
    idx.push(lastRing + i, lastRing + n, apex);
  }

  const triCount = idx.length / 3;
  console.log(`[geometry] cap: ${triCount} tris, ${ringLen} verts/ring, ${domeSegments + 1} rings`);
  return { positions: new Float32Array(pos), normals: new Float32Array(nrm), indices: new Uint16Array(idx) };
}

// ── LOD cross-section geometries ─────────────────────────────────────────

export const LOD_BODY_GEO: GeometryData[] = [
  // LOD 0: full detail (12-vertex rounded rectangle, current default)
  defaultCrossSectionGeometry(0.35, 5),
  // LOD 1: medium (8-vertex hex approximation — edgeSegments=3 prevents
  // opposing normals on adjacent vertices that create zero-crossings)
  defaultCrossSectionGeometry(0.35, 3),
  // LOD 2: flat quad (4-vertex rectangle)
  flatCrossSectionGeometry(0.35),
];

export const LOD_CAP_GEO: GeometryData[] = [
  // LOD 0: full caps
  defaultEndcapGeometry(0.35, 5, 2),
  // LOD 1: reduced caps (fewer dome segments)
  defaultEndcapGeometry(0.35, 3, 1),
  // LOD 2: no caps (skipped at far distance)
  { positions: new Float32Array(0), normals: new Float32Array(0), indices: new Uint16Array(0) },
];

function flatCrossSectionGeometry(hScale = 0.35): GeometryData {
  const R = hScale / 2;
  const W = 0.5;
  const cxL = -(W - R);
  const cxR =  W - R;

  // 4 corners of the rectangle
  const ringLen = 4;
  // Order: top-left, top-right, bottom-right, bottom-left
  const px = [cxL, cxR, cxR, cxL];
  const py = [ R,   R,  -R,  -R];

  // Normals point radially from center (smooth interpolation → cylinder-like)
  const nrm2: number[] = [];
  for (let i = 0; i < ringLen; i++) {
    const len = Math.sqrt(px[i] * px[i] + py[i] * py[i]);
    nrm2.push(px[i] / len, py[i] / len);
  }

  const posArr: number[] = [];
  const nrmArr: number[] = [];
  const idx: number[] = [];
  const nBody = 2; // fewer subdivisions for flat

  const e0 = 0;
  for (let i = 0; i < ringLen; i++) {
    posArr.push(px[i], py[i], -0.5);
    nrmArr.push(nrm2[i * 2], nrm2[i * 2 + 1], 0);
  }
  const e1 = ringLen;
  for (let i = 0; i < ringLen; i++) {
    posArr.push(px[i], py[i], 0.5);
    nrmArr.push(nrm2[i * 2], nrm2[i * 2 + 1], 0);
  }
  const body0 = 2 * ringLen;
  for (let j = 0; j < nBody; j++) {
    const z = -0.5 + (j + 1) / (nBody + 1);
    for (let i = 0; i < ringLen; i++) {
      posArr.push(px[i], py[i], z);
      nrmArr.push(nrm2[i * 2], nrm2[i * 2 + 1], 0);
    }
  }

  function cr(r0: number, r1: number) {
    for (let i = 0; i < ringLen; i++) {
      const n = (i + 1) % ringLen;
      idx.push(r0 + i, r1 + i, r1 + n);
      idx.push(r0 + i, r1 + n, r0 + n);
    }
  }
  cr(e0, body0);
  for (let j = 0; j < nBody - 1; j++)
    cr(body0 + j * ringLen, body0 + (j + 1) * ringLen);
  cr(body0 + (nBody - 1) * ringLen, e1);

  const triCount = idx.length / 3;
  console.log(`[geometry] LOD2 body: ${triCount} tris, ${ringLen} verts/ring, ${nBody + 2} rings`);
  return { positions: new Float32Array(posArr), normals: new Float32Array(nrmArr), indices: new Uint16Array(idx) };
}
