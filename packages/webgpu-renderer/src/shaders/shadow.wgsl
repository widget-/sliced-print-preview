// ── Shadow map generation (depth-only) ──
// Geometry MUST match the main vertex shader (segment.wgsl) exactly so the
// depth map correctly represents the visible geometry for shadow comparisons.

@group(0) @binding(0) var<uniform> shadowVP: mat4x4<f32>;
@group(0) @binding(1) var<storage, read> segments: array<f32>;

@vertex
fn vs_shadow(@builtin(instance_index) ii: u32, @location(0) pos: vec3<f32>, @location(1) nrm: vec3<f32>) -> @builtin(position) vec4<f32> {
  let stride: u32 = 16u; // 64 bytes / 4
  let base: u32 = ii * stride;

  let startPos: vec3<f32> = vec3<f32>(segments[base], segments[base + 1u], segments[base + 2u]);
  let width: f32 = segments[base + 3u];
  let endPos: vec3<f32> = vec3<f32>(segments[base + 4u], segments[base + 5u], segments[base + 6u]);

  // Along-segment position: match main shader's
  //   t = pos.z + 0.5;  segPos = mix(startPos, endPos, t)
  let t: f32 = pos.z + 0.5;
  let segPos: vec3<f32> = mix(startPos, endPos, t);

  // Build orthonormal basis from the segment direction (match main shader)
  let tangent: vec3<f32> = normalize(endPos - startPos);
  let upDir: vec3<f32> = vec3<f32>(0.0, 0.0, 1.0);
  var rightDir: vec3<f32> = -normalize(cross(upDir, tangent));
  if (length(rightDir) < 0.001) {
    rightDir = vec3<f32>(1.0, 0.0, 0.0);
  }

  // Match main shader geometry exactly: areaCorrection=1.1, hScale=1.25
  let hScale: f32 = 1.25;
  let areaCorrection: f32 = 1.1;
  let worldPos: vec3<f32> = segPos
    + rightDir * pos.x * width * areaCorrection
    + upDir * pos.y * width * hScale;
  return shadowVP * vec4<f32>(worldPos, 1.0);
}

@fragment
fn fs_shadow() {}
