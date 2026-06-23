/**
 * WGSL buffer layout for per-segment data.
 *
 * Using vec4<f32>-only struct for unambiguous 16-byte alignment.
 * Total: 64 bytes per segment (4 × vec4<f32>).
 */
export const SEGMENT_DATA_STRIDE = 64;

export const SEGMENT_DATA_OFFSET = {
  /** vec4<f32>: startX, startY, startZ, width */
  startPos: 0,
  /** vec4<f32>: endX, endY, endZ, conicWeight */
  endPos: 16,
  /** vec4<f32>: tanX, tanY, tanZ, layerZ */
  chainTangent: 32,
  /** vec4<f32>: pack0, _pad1, _pad2, _pad3 */
  pack0: 48,
} as const;

/** Packed-field bit layout for the `pack0` float. */
export const PACK = {
  SEGTYPE_BIT:      1 << 0,
  STARTCAP_BIT:     1 << 1,
  ENDCAP_BIT:       1 << 2,
  ROLE_SHIFT: 8,
  ROLE_MASK:  0xFF,
} as const;

export function pack0(segType: number, startCap: number, endCap: number, role: number): number {
  let v = 0;
  if (segType) v |= PACK.SEGTYPE_BIT;
  if (startCap) v |= PACK.STARTCAP_BIT;
  if (endCap) v |= PACK.ENDCAP_BIT;
  v |= (role & PACK.ROLE_MASK) << PACK.ROLE_SHIFT;
  return v;
}
