/**
 * WGSL buffer layout for per-segment data.
 *
 * Each segment is 48 bytes (3 × 16-byte aligned rows).
 * These constants let the JS side pack data into the same layout
 * the WGSL shader's SegmentData struct expects.
 */
export const SEGMENT_DATA_STRIDE = 48;

export const SEGMENT_DATA_OFFSET = {
  startPos: 0,        // vec3<f32> — bytes 0-11
  width: 12,          // f32       — bytes 12-15
  endPos: 16,         // vec3<f32> — bytes 16-27
  pack0: 28,          // f32       — bytes 28-31 (bitfield)
  chainTangent: 32,   // vec3<f32> — bytes 32-43
  layerZ: 44,         // f32       — bytes 44-47
} as const;

/** Packed-field bit layout for the `pack0` float. */
export const PACK = {
  SEGTYPE_BIT:      1 << 0,  // 0 = linear, 1 = arc
  STARTCAP_BIT:     1 << 1,  // segment is the start of a chain
  ENDCAP_BIT:       1 << 2,  // segment is the end of a chain
  ROLE_SHIFT: 8,
  ROLE_MASK:  0xFF,          // 8 bits for role (0–13)
  CONIC_WEIGHT_SHIFT: 16,
  CONIC_WEIGHT_MASK: 0xFFFF, // 16 bits fixed-point: conicWeight * 10000
} as const;

export function pack0(segType: number, startCap: number, endCap: number, role: number, conicWeight?: number): number {
  let v = 0;
  if (segType) v |= PACK.SEGTYPE_BIT;
  if (startCap) v |= PACK.STARTCAP_BIT;
  if (endCap) v |= PACK.ENDCAP_BIT;
  v |= (role & PACK.ROLE_MASK) << PACK.ROLE_SHIFT;
  if (conicWeight !== undefined) {
    v |= (Math.round(conicWeight * 10000) & PACK.CONIC_WEIGHT_MASK) << PACK.CONIC_WEIGHT_SHIFT;
  }
  return v;
}
