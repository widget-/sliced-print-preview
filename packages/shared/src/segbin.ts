import type { SegbinData } from './types';

export async function loadSegbin(url: string): Promise<SegbinData> {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`Failed to fetch ${url}: ${resp.status}`);
  const buf = await resp.arrayBuffer();
  return parseSegbin(buf);
}

export function parseSegbin(buf: ArrayBuffer): SegbinData {
  const dv = new DataView(buf);

  const magic = dv.getUint32(0, true);
  if (magic !== 0x31474553) {
    throw new Error(`Bad segbin magic: 0x${magic.toString(16)}`);
  }
  const version = dv.getUint16(4, true);
  if (version !== 1) {
    throw new Error(`Unsupported segbin version: ${version}`);
  }
  const count = dv.getUint32(8, true);
  const flags = dv.getUint32(12, true);

  const geoms = new Float32Array(buf, 16, count * 8);
  let offset = 16 + count * 32;
  const roles = new Uint8Array(buf, offset, count);
  offset += count;

  let chainContinue: Uint8Array;
  if (flags & 4) {
    chainContinue = new Uint8Array(buf, offset, count);
    offset += count;
  } else {
    chainContinue = new Uint8Array(count);
  }

  let segType: Uint8Array;
  if (flags & 8) {
    segType = new Uint8Array(buf, offset, count);
  } else {
    segType = new Uint8Array(count);
  }

  const zSet = new Set<number>();
  for (let i = 0; i < count; i++) {
    zSet.add(geoms[i * 8 + 7]);
  }
  const layerZs = Float32Array.from(zSet).sort();

  return { count, geoms, roles, chainContinue, segType, layerZs };
}
