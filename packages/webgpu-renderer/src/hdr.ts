/**
 * Minimal Radiance RGBE (.hdr) parser.
 * Returns a flat Float16Array of RGBA pixels, row-major, top-left origin.
 */
export interface HDRImage {
  width: number;
  height: number;
  exposure: number;
  data: Float16Array; // 4 channels per pixel (RGBA), row-major
}

export async function loadHDR(url: string): Promise<HDRImage> {
  const resp = await fetch(url);
  const buf = new Uint8Array(await resp.arrayBuffer());
  const len = buf.length;
  let off = 0;

  function nextLine(): string {
    let s = '';
    while (off < len) { const c = buf[off++]; if (c === 10) break; s += String.fromCharCode(c); }
    return s;
  }

  let width = 0, height = 0, exposure = 1;

  for (let i = 0; i < 32; i++) {
    const line = nextLine();
    if (/^#\?RADIANCE/.test(line)) continue;
    if (/FORMAT=32-bit_rle_rgbe/.test(line)) { continue; }
    const exp = line.match(/EXPOSURE=\s*([\d.]+)/);
    if (exp) { exposure = +exp[1]; continue; }
    const res = line.match(/-Y\s+(\d+)\s+\+X\s+(\d+)/);
    if (res) { height = +res[1]; width = +res[2]; break; }
  }
  if (!width || !height) throw new Error('Invalid HDR header');

  const raw = new Uint8Array(width * height * 4);
  readPixelsRLE(buf, raw, off, width, height);

  const data = new Float16Array(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    const r = raw[i * 4], g = raw[i * 4 + 1], b = raw[i * 4 + 2], e = raw[i * 4 + 3];
    const scale = Math.pow(2, e - 128);
    data[i * 4] = (r / 255) * scale;
    data[i * 4 + 1] = (g / 255) * scale;
    data[i * 4 + 2] = (b / 255) * scale;
    data[i * 4 + 3] = 1;
  }
  return { width, height, exposure, data };
}

function readPixelsRLE(
  buf: Uint8Array, out: Uint8Array, startOff: number,
  w: number, h: number,
) {
  let off = startOff;
  const scanline = new Array<number>(4 * w);
  for (let row = 0; row < h; row++) {
    const hdr = [buf[off++], buf[off++], buf[off++], buf[off++]];
    if (hdr[0] !== 2 || hdr[1] !== 2 || (hdr[2] & 0x80)) {
      // Non-RLE — write this pixel then copy remaining raw
      const o = row * w * 4;
      out[o] = hdr[0]; out[o + 1] = hdr[1]; out[o + 2] = hdr[2]; out[o + 3] = hdr[3];
      const remaining = (w * h - row * w - 1) * 4;
      for (let k = 0; k < remaining; k++) out[o + 4 + k] = buf[off++];
      return;
    }
    const scanW = ((hdr[2] & 0xff) << 8) | (hdr[3] & 0xff);
    if (scanW !== w) throw new Error(`Scanline width mismatch: ${scanW} vs ${w}`);

    for (let chan = 0; chan < 4; chan++) {
      const base = chan * w;
      let ptr = base;
      while (ptr < base + w) {
        const a = buf[off++];
        if (a > 128) {
          let count = a - 128;
          const val = buf[off++];
          while (count--) scanline[ptr++] = val;
        } else {
          let count = a;
          scanline[ptr++] = buf[off++];
          while (--count > 0) scanline[ptr++] = buf[off++];
        }
      }
    }
    const o = row * w * 4;
    for (let i = 0; i < w; i++) {
      out[o + i * 4] = scanline[i];
      out[o + i * 4 + 1] = scanline[i + w];
      out[o + i * 4 + 2] = scanline[i + 2 * w];
      out[o + i * 4 + 3] = scanline[i + 3 * w];
    }
  }
}
