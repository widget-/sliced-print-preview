import type { OrbitCamera } from './camera';
import type { SegbinData } from '@sliced/shared';

/**
 * CPU-side LOD evaluation.
 * Replicates the cull.wgsl compute shader logic.
 *
 * For each segment, computes the screen-space size of its bounding-box center
 * projected through the view-projection matrix, then assigns LOD:
 *   - screenSize < 0.5px → culled (not rendered)
 *   - screenSize < 5px  → LOD 2 (flat quad, no caps)
 *   - screenSize < 20px → LOD 1 (reduced body + caps)
 *   - otherwise          → LOD 0 (full body + domed caps)
 *
 * Arc segments inherit LOD from the previous segment.
 *
 * @returns An array of 3 Float32Arrays containing segment indices per LOD level.
 *          (lod0Indices, lod1Indices, lod2Indices)
 */
export function evaluateLOD(
  data: SegbinData,
  camera: OrbitCamera,
  canvasWidth: number,
  canvasHeight: number,
): [Float32Array, Float32Array, Float32Array] {
  const count = data.count;
  const g = data.geoms;
  const st = data.segType;

  const vp = camera.viewProj; // column-major mat4
  const screenDiag = Math.sqrt(canvasWidth * canvasWidth + canvasHeight * canvasHeight);

  // Temporary per-segment LOD
  const lod = new Uint8Array(count);

  for (let i = 0; i < count; i++) {
    const width = g[i * 8 + 6];
    if (width <= 0) {
      lod[i] = 255; // hidden segments → cull
      continue;
    }

    const sx = g[i * 8], sy = g[i * 8 + 1], sz = g[i * 8 + 2];
    const ex = g[i * 8 + 3], ey = g[i * 8 + 4], ez = g[i * 8 + 5];

    // Project a world-space point to NDC via VP matrix, return [ndcX, ndcY, ndcZ, w]
    const project = (px: number, py: number, pz: number): [number, number, number, number] => {
      const cx = vp[0] * px + vp[4] * py + vp[8] * pz + vp[12];
      const cy = vp[1] * px + vp[5] * py + vp[9] * pz + vp[13];
      const cz = vp[2] * px + vp[6] * py + vp[10] * pz + vp[14];
      const cw = vp[3] * px + vp[7] * py + vp[11] * pz + vp[15];
      if (Math.abs(cw) < 0.0001) return [0, 0, -1, 0];
      return [cx / cw, cy / cw, cz / cw, cw];
    };

    // Test both endpoints — keep segment if either is visible
    const [ndcSX, ndcSY, ndcSZ] = project(sx, sy, sz);
    const [ndcEX, ndcEY, ndcEZ] = project(ex, ey, ez);
    const [ndcMX, ndcMY, ndcMZ, wM] = project((sx + ex) / 2, (sy + ey) / 2, (sz + ez) / 2);

    const inFrustum = (nx: number, ny: number, nz: number) =>
      Math.abs(nx) <= 1.1 && Math.abs(ny) <= 1.1 && nz >= -0.1 && nz <= 1.1;

    const startIn = inFrustum(ndcSX, ndcSY, ndcSZ);
    const endIn = inFrustum(ndcEX, ndcEY, ndcEZ);
    const midIn = inFrustum(ndcMX, ndcMY, ndcMZ);

    if (!startIn && !endIn && !midIn) {
      lod[i] = 255; // culled
      continue;
    }

    // Use midpoint's w for screen-size computation
    const segLen = Math.sqrt((ex - sx) ** 2 + (ey - sy) ** 2 + (ez - sz) ** 2);
    const visSize = Math.max(segLen, width);
    const dist = Math.abs(wM);
    const screenSize = (visSize / Math.max(dist, 0.001)) * screenDiag * 0.5;

    if (screenSize < 0.5) {
      lod[i] = 255; // too small → cull
    } else if (screenSize < 5) {
      lod[i] = 2;
    } else if (screenSize < 20) {
      lod[i] = 1;
    } else {
      lod[i] = 0;
    }
  }

  // Arc LOD inheritance: arcs get the LOD of the previous segment
  // (but culled predecessors produce culled arcs)
  for (let i = 1; i < count; i++) {
    if (st[i] === 1) {
      lod[i] = lod[i - 1];
    }
  }

  // Count per-LOD and build index arrays (skip culled: 255)
  const counts = [0, 0, 0];
  for (let i = 0; i < count; i++) {
    if (lod[i] === 255) continue;
    counts[lod[i]]++;
  }

  const lod0 = new Float32Array(counts[0]);
  const lod1 = new Float32Array(counts[1]);
  const lod2 = new Float32Array(counts[2]);
  const cursors = [0, 0, 0];

  for (let i = 0; i < count; i++) {
    const l = lod[i];
    if (l === 0) lod0[cursors[0]++] = i;
    else if (l === 1) lod1[cursors[1]++] = i;
    else if (l === 2) lod2[cursors[2]++] = i;
    // 255 = culled, skip
  }

  return [lod0, lod1, lod2];
}
