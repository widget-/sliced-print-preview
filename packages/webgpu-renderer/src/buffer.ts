import { SEGMENT_DATA_STRIDE, SEGMENT_DATA_OFFSET, pack0 } from './wgsl-types';
import { roleColor } from '@sliced/shared';
import type { SegbinData } from '@sliced/shared';

export interface GpuSegmentBuffers {
  /** Storage buffer containing array<SegmentData> for the vertex shader. */
  segmentBuffer: GPUBuffer;
  /** Storage buffer containing array<vec4<f32>> RGBA colors, one per segment. */
  colorBuffer: GPUBuffer;
  /** Number of segments in the buffer. */
  count: number;
}

/**
 * Repack parsed segbin data into GPU storage buffers.
 *
 * For each segment, computes chain metadata (start/end caps, chain-start tangent)
 * and packs everything into WGSL-friendly 48-byte structs + a separate color buffer.
 */
export function buildSegmentBuffers(device: GPUDevice, data: SegbinData): GpuSegmentBuffers {
  const count = data.count;
  const g = data.geoms;
  const cc = data.chainContinue;
  const st = data.segType;
  const roles = data.roles;

  // ── Repack into 48-byte aligned structs ──
  const bufSize = count * SEGMENT_DATA_STRIDE;
  const segmentBuf = device.createBuffer({
    size: bufSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  const view = new DataView(segmentBuf.getMappedRange());

  for (let i = 0; i < count; i++) {
    const base = i * SEGMENT_DATA_STRIDE;
    const sx = g[i * 8];     const sy = g[i * 8 + 1]; const sz = g[i * 8 + 2];
    const ex = g[i * 8 + 3]; const ey = g[i * 8 + 4]; const ez = g[i * 8 + 5];
    const layerZ = g[i * 8 + 7];
    const isArc = st[i] === 1;
    let conicWeight: number | undefined;

    if (isArc) {
      const arcWidth = g[i * 8 + 6];
      const packed = g[i * 8 + 7];
      const lz = Math.round(packed * 100) / 100;
      conicWeight = (packed - lz) * 10000;
      view.setFloat32(base + SEGMENT_DATA_OFFSET.startPos, sx, true);
      view.setFloat32(base + SEGMENT_DATA_OFFSET.startPos + 4, sy, true);
      view.setFloat32(base + SEGMENT_DATA_OFFSET.startPos + 8, sz, true);
      view.setFloat32(base + SEGMENT_DATA_OFFSET.width, arcWidth, true);
      view.setFloat32(base + SEGMENT_DATA_OFFSET.endPos, ex, true);
      view.setFloat32(base + SEGMENT_DATA_OFFSET.endPos + 4, ey, true);
      view.setFloat32(base + SEGMENT_DATA_OFFSET.endPos + 8, ez, true);
      view.setFloat32(base + SEGMENT_DATA_OFFSET.layerZ, lz, true);
    } else {
      const width = g[i * 8 + 6];
      view.setFloat32(base + SEGMENT_DATA_OFFSET.startPos, sx, true);
      view.setFloat32(base + SEGMENT_DATA_OFFSET.startPos + 4, sy, true);
      view.setFloat32(base + SEGMENT_DATA_OFFSET.startPos + 8, sz, true);
      view.setFloat32(base + SEGMENT_DATA_OFFSET.width, width, true);
      view.setFloat32(base + SEGMENT_DATA_OFFSET.endPos, ex, true);
      view.setFloat32(base + SEGMENT_DATA_OFFSET.endPos + 4, ey, true);
      view.setFloat32(base + SEGMENT_DATA_OFFSET.endPos + 8, ez, true);
      view.setFloat32(base + SEGMENT_DATA_OFFSET.layerZ, layerZ, true);
    }

    // Chain metadata
    const startCap = (i === 0 || cc[i - 1] === 0) ? 1 : 0;
    const endCap   = (i === count - 1 || cc[i] === 0) ? 1 : 0;

    // Chain-start tangent
    let ctx: number, cty: number, ctz: number;
    if (i > 0 && cc[i - 1] === 1) {
      if (st[i - 1] === 1) {
        const p1x = g[(i-1)*8 + 3], p1y = g[(i-1)*8 + 4], p1z = g[(i-1)*8 + 5];
        const p2x = g[i*8], p2y = g[i*8 + 1], p2z = g[i*8 + 2];
        const dx = p2x - p1x, dy = p2y - p1y, dz = p2z - p1z;
        const len = Math.sqrt(dx*dx + dy*dy + dz*dz);
        if (len > 0.001) { ctx = dx/len; cty = dy/len; ctz = dz/len; }
        else { const d = segDir(g, i); ctx = d[0]; cty = d[1]; ctz = d[2]; }
      } else {
        const d = segDir(g, i - 1); ctx = d[0]; cty = d[1]; ctz = d[2];
      }
    } else {
      const d = segDir(g, i); ctx = d[0]; cty = d[1]; ctz = d[2];
    }

    view.setFloat32(base + SEGMENT_DATA_OFFSET.chainTangent, ctx, true);
    view.setFloat32(base + SEGMENT_DATA_OFFSET.chainTangent + 4, cty, true);
    view.setFloat32(base + SEGMENT_DATA_OFFSET.chainTangent + 8, ctz, true);

    // Pack flags
    const packed = pack0(isArc ? 1 : 0, startCap, endCap, roles[i], conicWeight);
    view.setFloat32(base + SEGMENT_DATA_OFFSET.pack0, packed, true);
  }

  segmentBuf.unmap();

  // ── Color buffer ──
  const colorSize = count * 16; // vec4<f32> per segment
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

  return { segmentBuffer: segmentBuf, colorBuffer: colorBuf, count };
}

function segDir(g: Float32Array, i: number): [number, number, number] {
  const dx = g[i * 8 + 3] - g[i * 8], dy = g[i * 8 + 4] - g[i * 8 + 1], dz = g[i * 8 + 5] - g[i * 8 + 2];
  const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
  if (len < 0.001) return [0, 0, 1];
  return [dx / len, dy / len, dz / len];
}
