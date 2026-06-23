/** Convert Rust Duration Debug (e.g. "1s", "185ms", "902µs") → ms. */
export function parseDur(s: string): number | undefined {
  const m = s.match(/^([\d.]+)(s|ms|µs)$/);
  if (!m) return undefined;
  const v = parseFloat(m[1]!);
  if (m[2] === 's') return v * 1000;
  if (m[2] === 'ms') return v;
  return v * 0.001; // µs
}

/** Parse gcode-to-segbin stderr into structured timing. */
export function parseG2sTiming(stderr: string): Record<string, number> {
  const t: Record<string, number> = {};
  const m1 = stderr.match(/Parsed .+ \([\d.]+ merged, ([\d.]+)ms\)/);
  if (m1) t.parse = parseFloat(m1[1]!);

  const m2 = stderr.match(/Ray cull: (\d+) rays, .+ culled \(ray=([^,]+), seg_bvh=([^,]+), gap=([^)]+)\)/);
  if (m2) {
    t.rays = parseInt(m2[1]!);
    const r = parseDur(m2[2]!); if (r !== undefined) t.ray = r;
    const s = parseDur(m2[3]!); if (s !== undefined) t.segBvh = s;
    const g = parseDur(m2[4]!); if (g !== undefined) t.gap = g;
  }

  const m3 = stderr.match(/Arc subdivision: .+ \(([^)]+)\)/);
  if (m3) { const a = parseDur(m3[1]!); if (a !== undefined) t.arc = a; }

  const m4 = stderr.match(/Total: ([\d.]+[a-zµ]+)/);
  if (m4) { const u = parseDur(m4[1]!); if (u !== undefined) t.total = u; }

  return t;
}
