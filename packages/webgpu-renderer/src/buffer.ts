import { SEGMENT_DATA_STRIDE, SEGMENT_DATA_OFFSET, pack0 } from './wgsl-types';
import { roleColor, Role } from '@sliced/shared';
import type { SegbinData } from '@sliced/shared';

export interface GpuSegmentBuffers {
  segmentBuffer: GPUBuffer;
  colorBuffer: GPUBuffer;
  count: number;
  capBuffer: GPUBuffer;
  capCount: number;
}

/** Roles that shouldn't be rendered (same as WebGL2 HIDDEN_ROLES). */
export const HIDDEN_ROLES: Set<number> = new Set([Role.SkirtBrim, Role.Other]);

export function buildSegmentBuffers(
  device: GPUDevice,
  data: SegbinData,
  excludeRoles?: Set<number>,
): GpuSegmentBuffers {
  const count = data.count;
  const g = data.geoms;
  const cc = data.chainContinue;
  const st = data.segType;
  const roles = data.roles;

  const bufSize = count * SEGMENT_DATA_STRIDE;
  const segmentBuf = device.createBuffer({
    size: bufSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  const view = new DataView(segmentBuf.getMappedRange());
  for (let i = 0; i < count; i++) {
    const base = i * SEGMENT_DATA_STRIDE;
    const sx = g[i * 8],      sy = g[i * 8 + 1], sz = g[i * 8 + 2];
    const ex = g[i * 8 + 3], ey = g[i * 8 + 4], ez = g[i * 8 + 5];
    const isArc = st[i] === 1;

    // startPos: [sx, sy, sz, width]
    view.setFloat32(base + SEGMENT_DATA_OFFSET.startPos, sx, true);
    view.setFloat32(base + SEGMENT_DATA_OFFSET.startPos + 4, sy, true);
    view.setFloat32(base + SEGMENT_DATA_OFFSET.startPos + 8, sz, true);
    const rawWidth = g[i * 8 + 6];
    const effWidth = excludeRoles?.has(roles[i]) ? 0 : rawWidth;
    view.setFloat32(base + SEGMENT_DATA_OFFSET.startPos + 12, effWidth, true);

    // endPos: [ex, ey, ez, conicWeight]
    view.setFloat32(base + SEGMENT_DATA_OFFSET.endPos, ex, true);
    view.setFloat32(base + SEGMENT_DATA_OFFSET.endPos + 4, ey, true);
    view.setFloat32(base + SEGMENT_DATA_OFFSET.endPos + 8, ez, true);
    view.setFloat32(base + SEGMENT_DATA_OFFSET.endPos + 12, 0, true); // placeholder conicWeight

    if (isArc) {
      const packed = g[i * 8 + 7];
      const lz = Math.round(packed * 100) / 100;
      const cw = (packed - lz) * 10000;
      view.setFloat32(base + SEGMENT_DATA_OFFSET.endPos + 12, cw, true);
      view.setFloat32(base + SEGMENT_DATA_OFFSET.chainTangent + 12, lz, true);
    } else {
      view.setFloat32(base + SEGMENT_DATA_OFFSET.chainTangent + 12, g[i * 8 + 7], true);
    }

    // Chain-start tangent
    const chainedStart = i > 0 && (cc[i - 1] !== 0 || (isContinuous(g, i) && segmentsAligned(g, i)));
    const chainedEnd   = i < count - 1 && (cc[i] !== 0 || (isContinuous(g, i + 1) && segmentsAligned(g, i + 1)));
    const startCap = (i === 0 || !chainedStart) ? 1 : 0;
    const endCap   = (i === count - 1 || !chainedEnd) ? 1 : 0;

    let ctx: number, cty: number, ctz: number;
    if (i > 0 && st[i - 1] === 1) {
      const p1x = g[(i-1)*8 + 3], p1y = g[(i-1)*8 + 4], p1z = g[(i-1)*8 + 5];
      const p2x = g[i*8], p2y = g[i*8 + 1], p2z = g[i*8 + 2];
      const dx = p2x - p1x, dy = p2y - p1y, dz = p2z - p1z;
      const len = Math.sqrt(dx*dx + dy*dy + dz*dz);
      if (len > 0.001) { ctx = dx/len; cty = dy/len; ctz = dz/len; }
      else { const d = segDir(g, i); ctx = d[0]; cty = d[1]; ctz = d[2]; }
    } else if (i > 0 && cc[i - 1] === 1) {
      const d = segDir(g, i - 1); ctx = d[0]; cty = d[1]; ctz = d[2];
    } else {
      const d = segDir(g, i); ctx = d[0]; cty = d[1]; ctz = d[2];
    }

    view.setFloat32(base + SEGMENT_DATA_OFFSET.chainTangent, ctx, true);
    view.setFloat32(base + SEGMENT_DATA_OFFSET.chainTangent + 4, cty, true);
    view.setFloat32(base + SEGMENT_DATA_OFFSET.chainTangent + 8, ctz, true);

    view.setFloat32(base + SEGMENT_DATA_OFFSET.pack0, pack0(isArc ? 1 : 0, startCap, endCap, roles[i]), true);
  }

  segmentBuf.unmap();

  // ── Color buffer ──
  const colorSize = count * 16;
  const colorBuf = device.createBuffer({
    size: colorSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  const colorView = new Float32Array(colorBuf.getMappedRange());
  for (let i = 0; i < count; i++) {
    const hex = roleColor(roles[i]);
    colorView[i * 4] = parseInt(hex.slice(1, 3), 16) / 255;
    colorView[i * 4 + 1] = parseInt(hex.slice(3, 5), 16) / 255;
    colorView[i * 4 + 2] = parseInt(hex.slice(5, 7), 16) / 255;
    colorView[i * 4 + 3] = 1.0;
  }
  colorBuf.unmap();

  // ── Cap instance buffer ──
  let capCount = 0;
  for (let i = 0; i < count; i++) {
    const chainedStart = i > 0 && (cc[i - 1] !== 0 || (isContinuous(g, i) && segmentsAligned(g, i)));
    const chainedEnd   = i < count - 1 && (cc[i] !== 0 || (isContinuous(g, i + 1) && segmentsAligned(g, i + 1)));
    if (i === 0 || !chainedStart) capCount++;
    if (i === count - 1 || !chainedEnd) capCount++;
  }
  const capBuf = device.createBuffer({
    size: capCount * 8,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  const capView = new Float32Array(capBuf.getMappedRange());
  let capN = 0;
  for (let i = 0; i < count; i++) {
    const chainedStart = i > 0 && (cc[i - 1] !== 0 || (isContinuous(g, i) && segmentsAligned(g, i)));
    const chainedEnd   = i < count - 1 && (cc[i] !== 0 || (isContinuous(g, i + 1) && segmentsAligned(g, i + 1)));
    if (i === 0 || !chainedStart) {
      capView[capN * 2] = i; capView[capN * 2 + 1] = 0; capN++;
    }
    if (i === count - 1 || !chainedEnd) {
      capView[capN * 2] = i; capView[capN * 2 + 1] = 1; capN++;
    }
  }
  capBuf.unmap();

  return { segmentBuffer: segmentBuf, colorBuffer: colorBuf, count, capBuffer: capBuf, capCount };
}

function segDir(g: Float32Array, i: number): [number, number, number] {
  const dx = g[i * 8 + 3] - g[i * 8], dy = g[i * 8 + 4] - g[i * 8 + 1], dz = g[i * 8 + 5] - g[i * 8 + 2];
  const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
  if (len < 0.001) return [0, 0, 1];
  return [dx / len, dy / len, dz / len];
}

/** Check if two consecutive segments are geometrically continuous (same endpoint). */
function isContinuous(g: Float32Array, i: number): boolean {
  // Compares seg[i-1].end with seg[i].start
  const dx = g[(i - 1) * 8 + 3] - g[i * 8];
  const dy = g[(i - 1) * 8 + 4] - g[i * 8 + 1];
  const dz = g[(i - 1) * 8 + 5] - g[i * 8 + 2];
  return (dx * dx + dy * dy + dz * dz) < 0.0001;
}

/** Check that two segments meeting at a point go in the same general direction (angle < 90°).
 *  This prevents chaining loop-backs where path reverses direction at the same endpoint. */
function segmentsAligned(g: Float32Array, i: number): boolean {
  // seg[i-1] direction
  const aDx = g[(i - 1) * 8 + 3] - g[(i - 1) * 8];
  const aDy = g[(i - 1) * 8 + 4] - g[(i - 1) * 8 + 1];
  const aDz = g[(i - 1) * 8 + 5] - g[(i - 1) * 8 + 2];
  const aLen = Math.sqrt(aDx * aDx + aDy * aDy + aDz * aDz);
  if (aLen < 0.001) return true;
  // seg[i] direction
  const bDx = g[i * 8 + 3] - g[i * 8];
  const bDy = g[i * 8 + 4] - g[i * 8 + 1];
  const bDz = g[i * 8 + 5] - g[i * 8 + 2];
  const bLen = Math.sqrt(bDx * bDx + bDy * bDy + bDz * bDz);
  if (bLen < 0.001) return true;
  const dot = (aDx * bDx + aDy * bDy + aDz * bDz) / (aLen * bLen);
  return dot > 0; // positive dot = same hemisphere
}
