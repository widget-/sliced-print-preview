// ── Shadow map generation (depth-only) ──
// Geometry MUST match the main vertex shader (segment.wgsl) exactly so the
// depth map correctly represents the visible geometry for shadow comparisons.
// This includes proper arc (rational quadratic Bézier) handling.

@group(0) @binding(0) var<uniform> shadowVP: mat4x4<f32>;
@group(0) @binding(1) var<storage, read> segments: array<f32>;
@group(0) @binding(2) var<uniform> arcCurvature: vec4<f32>;  // x = curvature multiplier

@vertex
fn vs_shadow(@builtin(instance_index) ii: u32, @location(0) pos: vec3<f32>, @location(1) nrm: vec3<f32>) -> @builtin(position) vec4<f32> {
  let stride: u32 = 16u; // 64 bytes / 4
  let base: u32 = ii * stride;

  let startPos: vec3<f32> = vec3<f32>(segments[base], segments[base + 1u], segments[base + 2u]);
  let width: f32 = segments[base + 3u];
  let endPos: vec3<f32> = vec3<f32>(segments[base + 4u], segments[base + 5u], segments[base + 6u]);
  let conicWeight: f32 = segments[base + 7u]; // endPos.w for arcs
  let packed: u32 = u32(segments[base + 12u]); // misc.x
  let isArc: bool = (packed & 1u) != 0u;
  let endCapNeeded: bool = (packed & 4u) != 0u;
  let startCapNeeded: bool = (packed & 2u) != 0u;
  let baseWidth: f32 = width;
  let t: f32 = pos.z + 0.5;

  // Width interpolation at chained boundaries (matches segment.wgsl)
  var effectiveWidth: f32 = baseWidth;
  if (!endCapNeeded && t > 0.75) {
    let nextWidth = segments[base + 16u + 3u];
    if (abs(nextWidth - baseWidth) > 0.0001) {
      let frac = (t - 0.75) / 0.25;
      effectiveWidth = mix(baseWidth, nextWidth, frac);
    }
  }
  if (!startCapNeeded && t < 0.25 && ii > 0u) {
    let prevWidth = segments[base - 16u + 3u];
    if (abs(prevWidth - baseWidth) > 0.0001) {
      let frac = (0.25 - t) / 0.25;
      effectiveWidth = mix(prevWidth, baseWidth, frac);
    }
  }

  var segPos: vec3<f32>;
  var tangent: vec3<f32>;

  if (isArc) {
    // Rational quadratic Bézier arc matching segment.wgsl exactly
    let p0: vec3<f32> = startPos;
    let p1: vec3<f32> = endPos;
    let p2: vec3<f32> = vec3<f32>(segments[base + 16u], segments[base + 17u], segments[base + 18u]); // next segment's start
    let w: f32 = conicWeight * arcCurvature.x;
    let mt: f32 = 1.0 - t;
    let mt2: f32 = mt * mt;
    let t2: f32 = t * t;
    let denom: f32 = mt2 + 2.0 * t * mt * w + t2;
    segPos = (mt2 * p0 + 2.0 * t * mt * w * p1 + t2 * p2) / denom;

    // Finite-difference tangent (same as segment.wgsl)
    let eps: f32 = 0.01;
    let te: f32 = min(t + eps, 1.0); let me: f32 = 1.0 - te;
    let me2: f32 = me * me; let te2: f32 = te * te;
    let de: f32 = me2 + 2.0 * te * me * w + te2;
    let pe: vec3<f32> = (me2 * p0 + 2.0 * te * me * w * p1 + te2 * p2) / de;
    let ts: f32 = max(t - eps, 0.0); let ms: f32 = 1.0 - ts;
    let ms2: f32 = ms * ms; let ts2: f32 = ts * ts;
    let ds: f32 = ms2 + 2.0 * ts * ms * w + ts2;
    let ps: vec3<f32> = (ms2 * p0 + 2.0 * ts * ms * w * p1 + ts2 * p2) / ds;
    let dDir: vec3<f32> = pe - ps;
    let dLen: f32 = length(dDir);
    tangent = select(normalize(dDir), vec3<f32>(0.0, 0.0, 1.0), dLen < 0.0001);
  } else {
    segPos = mix(startPos, endPos, t);
    let dir: vec3<f32> = endPos - startPos;
    let segLen: f32 = length(dir);
    tangent = select(dir / segLen, vec3<f32>(0.0, 0.0, 1.0), segLen < 0.001);
  }

  // Build orthonormal basis from the segment direction (match main shader)
  let upDir: vec3<f32> = vec3<f32>(0.0, 0.0, 1.0);
  var rightDir: vec3<f32> = -normalize(cross(upDir, tangent));
  if (length(rightDir) < 0.001) {
    rightDir = vec3<f32>(1.0, 0.0, 0.0);
  }

  // Match main shader geometry exactly: areaCorrection=1.1, hScale=1.25
  let hScale: f32 = 1.25;
  let areaCorrection: f32 = 1.1;
  let worldPos: vec3<f32> = segPos
    + rightDir * pos.x * effectiveWidth * areaCorrection
    + upDir * pos.y * effectiveWidth * hScale;
  return shadowVP * vec4<f32>(worldPos, 1.0);
}

@fragment
fn fs_shadow() {}
