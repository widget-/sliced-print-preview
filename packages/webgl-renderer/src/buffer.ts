import { roleColor, Role } from '@sliced/shared';
import type { SegbinData } from '@sliced/shared';

export interface SegmentTextures {
  /** Float32 RGBA texture: 4 texels per segment. */
  segTex: WebGLTexture;
  /** Float32 RGBA texture: 1 texel per segment (role color). */
  colorTex: WebGLTexture;
  /** Width of both textures in texels. */
  texWidth: number;
  /** Height of segTex in texels. */
  segTexHeight: number;
  /** Height of colorTex in texels. */
  colorTexHeight: number;
  /** Total segment count. */
  count: number;
  /** Per-segment LOD data — computed each frame on CPU. */
  segmentLod: Uint8Array;
  /** Cap instance data: flat [segIdx, isEnd] pairs. */
  capInstances: Float32Array;
  capCount: number;
}

/** Roles that shouldn't be rendered. */
export const HIDDEN_ROLES: Set<number> = new Set([Role.SkirtBrim, Role.Other]);

/** Pack flags into a single float (same bit layout as WGSL pack0). */
function pack0(segType: number, startCap: number, endCap: number, role: number): number {
  let v = 0;
  if (segType) v |= 1;
  if (startCap) v |= 2;
  if (endCap) v |= 4;
  v |= (role & 0xFF) << 8;
  return v;
}

function segDir(g: Float32Array, i: number): [number, number, number] {
  const dx = g[i * 8 + 3] - g[i * 8];
  const dy = g[i * 8 + 4] - g[i * 8 + 1];
  const dz = g[i * 8 + 5] - g[i * 8 + 2];
  const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
  if (len < 0.001) return [0, 0, 1];
  return [dx / len, dy / len, dz / len];
}

export function buildSegmentTextures(
  gl: WebGL2RenderingContext,
  data: SegbinData,
  excludeRoles?: Set<number>,
): SegmentTextures {
  const count = data.count;
  const g = data.geoms;
  const cc = data.chainContinue;
  const st = data.segType;
  const roles = data.roles;

  // ── Determine texture dimensions ──
  const maxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE);
  const texWidth = Math.min(2048, maxTex);

  // Segment texture: 4 texels per segment
  const segTexels = count * 4;
  const segTexHeight = Math.ceil(segTexels / texWidth);
  const segData = new Float32Array(texWidth * segTexHeight * 4); // RGBA

  // Color texture: 1 texel per segment
  const colorTexels = count;
  const colorTexHeight = Math.ceil(colorTexels / texWidth);
  const colorData = new Float32Array(texWidth * colorTexHeight * 4);

  // ── Pack segment data ──
  for (let i = 0; i < count; i++) {
    const sx = g[i * 8], sy = g[i * 8 + 1], sz = g[i * 8 + 2];
    const ex = g[i * 8 + 3], ey = g[i * 8 + 4], ez = g[i * 8 + 5];
    const isArc = st[i] === 1;
    const rawWidth = g[i * 8 + 6];
    const effWidth = excludeRoles?.has(roles[i]) ? 0 : rawWidth;

    // Start cap flag
    const startCap = (i === 0 || cc[i - 1] === 0) ? 1 : 0;
    const endCap = (i === count - 1 || cc[i] === 0) ? 1 : 0;

    // Conic weight for arcs
    let conicWeight = 0;
    let layerZ = g[i * 8 + 7];
    if (isArc) {
      const packed = g[i * 8 + 7];
      layerZ = Math.round(packed * 100) / 100;
      conicWeight = (packed - layerZ) * 10000;
    }

    // Chain-start tangent
    let ctx: number, cty: number, ctz: number;
    if (i > 0 && st[i - 1] === 1) {
      const p1x = g[(i - 1) * 8 + 3], p1y = g[(i - 1) * 8 + 4], p1z = g[(i - 1) * 8 + 5];
      const p2x = g[i * 8], p2y = g[i * 8 + 1], p2z = g[i * 8 + 2];
      const dx = p2x - p1x, dy = p2y - p1y, dz = p2z - p1z;
      const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (len > 0.001) { ctx = dx / len; cty = dy / len; ctz = dz / len; }
      else { const d = segDir(g, i); ctx = d[0]; cty = d[1]; ctz = d[2]; }
    } else if (i > 0 && cc[i - 1] === 1) {
      const d = segDir(g, i - 1); ctx = d[0]; cty = d[1]; ctz = d[2];
    } else {
      const d = segDir(g, i); ctx = d[0]; cty = d[1]; ctz = d[2];
    }

    // Write 4 texels for this segment
    const t0 = i * 4;     // startPos
    const t1 = i * 4 + 1; // endPos
    const t2 = i * 4 + 2; // chain tangent + layerZ
    const t3 = i * 4 + 3; // pack0

    const writeTexel = (texelIdx: number, r: number, g: number, b: number, a: number) => {
      const col = texelIdx % texWidth;
      const row = Math.floor(texelIdx / texWidth);
      const off = (row * texWidth + col) * 4;
      segData[off] = r;
      segData[off + 1] = g;
      segData[off + 2] = b;
      segData[off + 3] = a;
    };

    writeTexel(t0, sx, sy, sz, effWidth);
    writeTexel(t1, ex, ey, ez, conicWeight);
    writeTexel(t2, ctx, cty, ctz, layerZ);
    writeTexel(t3, pack0(isArc ? 1 : 0, startCap, endCap, roles[i]), 0, 0, 0);

    // Color texel
    const hex = roleColor(roles[i]);
    const cCol = i % texWidth;
    const cRow = Math.floor(i / texWidth);
    const cOff = (cRow * texWidth + cCol) * 4;
    colorData[cOff] = parseInt(hex.slice(1, 3), 16) / 255;
    colorData[cOff + 1] = parseInt(hex.slice(3, 5), 16) / 255;
    colorData[cOff + 2] = parseInt(hex.slice(5, 7), 16) / 255;
    colorData[cOff + 3] = 1.0;
  }

  // ── Upload segment texture ──
  const segTex = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, segTex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, texWidth, segTexHeight, 0, gl.RGBA, gl.FLOAT, segData);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

  // ── Upload color texture ──
  const colorTex = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, colorTex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, texWidth, colorTexHeight, 0, gl.RGBA, gl.FLOAT, colorData);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

  gl.bindTexture(gl.TEXTURE_2D, null);

  // ── Cap instances ──
  let capCount = 0;
  for (let i = 0; i < count; i++) {
    if (i === 0 || cc[i - 1] === 0) capCount++;
    if (i === count - 1 || cc[i] === 0) capCount++;
  }
  const capInstances = new Float32Array(capCount * 2);
  let capN = 0;
  for (let i = 0; i < count; i++) {
    if (i === 0 || cc[i - 1] === 0) {
      capInstances[capN * 2] = i;
      capInstances[capN * 2 + 1] = 0;
      capN++;
    }
    if (i === count - 1 || cc[i] === 0) {
      capInstances[capN * 2] = i;
      capInstances[capN * 2 + 1] = 1;
      capN++;
    }
  }

  // ── LOD data (initialized to 0, recomputed each frame) ──
  const segmentLod = new Uint8Array(count);

  return { segTex, colorTex, texWidth, segTexHeight, colorTexHeight, count, segmentLod, capInstances, capCount };
}
