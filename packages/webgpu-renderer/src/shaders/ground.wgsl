// ── Ground plane shader ──
//
// Renders a single-sided ground plane at z = groundZ (below the model).
// The plane is visible from above (+Z direction) and invisible from below.
//
// Shadow pass:  writes depth only (fs_ground_shadow)
// Main pass:    outputs a ground color modulated by the shadow map (fs_ground)
//
// Bind groups (main pass):
//   group(0) @binding(0): camera (viewProj) — for world→clip transform
//   group(1) @binding(0): shadowTex (depth)
//   group(1) @binding(1): shadowSampler (comparison)
//   group(1) @binding(2): shadowVP (mat4x4) — for shadow UV computation
//
// See types.wgsl for the Camera struct and shadow bindings.

struct Camera {
  viewProj: mat4x4<f32>,
  viewMat: mat4x4<f32>,
};

struct VSIn {
  @location(0) position: vec3<f32>,
};

struct VSOut {
  @builtin(position) clipPos: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;

@vertex
fn vs_ground(in: VSIn) -> VSOut {
  var out: VSOut;
  let worldPos: vec3<f32> = in.position;
  out.clipPos = camera.viewProj * vec4<f32>(worldPos, 1.0);
  out.worldPos = worldPos;
  return out;
}

// ── Shadow pass: depth only ──
// No color output; depth is written automatically by rasterization.
// Fragment shader is a stub (required by WebGPU even for depth-only passes).

struct VSOutShadow {
  @builtin(position) clipPos: vec4<f32>,
};

@group(0) @binding(0) var<uniform> shadowVPMat: mat4x4<f32>;

@vertex
fn vs_ground_shadow(in: VSIn) -> VSOutShadow {
  var out: VSOutShadow;
  out.clipPos = shadowVPMat * vec4<f32>(in.position, 1.0);
  return out;
}

@fragment
fn fs_ground_shadow() {}

// ── Main pass: ground color with shadow ──
// Uses the same shadow map bindings as the PBR shader (group 1).

@group(1) @binding(0) var shadowTex: texture_depth_2d;
@group(1) @binding(1) var shadowSampler: sampler_comparison;
@group(1) @binding(2) var<uniform> shadowVP: mat4x4<f32>;

struct FragOut {
  @location(0) color: vec4<f32>,
  @location(1) normal: vec4<f32>,
};

@fragment
fn fs_ground(in: VSOut) -> FragOut {
  // Ground color (subtle warm gray, like a concrete studio floor)
  let groundColor: vec3<f32> = vec3<f32>(0.35, 0.33, 0.30);

  // Shadow map lookup (same PCF 5×5 as pbr.wgsl)
  let shadowClip: vec4<f32> = shadowVP * vec4<f32>(in.worldPos, 1.0);
  let shadowNDC: vec3<f32> = shadowClip.xyz / shadowClip.w;
  let shadowUV: vec2<f32> = shadowNDC.xy * vec2<f32>(0.5, -0.5) + 0.5;
  let texel: f32 = 1.0 / 1024.0;
  let bias: f32 = 0.002;
  var shadowVis: f32 = 0.0;
  for (var dy: i32 = -2; dy <= 2; dy++) {
    for (var dx: i32 = -2; dx <= 2; dx++) {
      let uv: vec2<f32> = shadowUV + vec2<f32>(f32(dx), f32(dy)) * texel;
      shadowVis += textureSampleCompareLevel(shadowTex, shadowSampler, uv, shadowNDC.z - bias);
    }
  }
  shadowVis /= 25.0;

  // Ambient fill: soften the shadow so the ground isn't pure black in shadow
  let ambientFill: f32 = 0.15;
  let lit: f32 = max(ambientFill, shadowVis);

  // Write a normal facing +Z (upward) so SSAO at the ground/model boundary is consistent
  let worldN: vec3<f32> = vec3<f32>(0.0, 0.0, 1.0);
  let viewN: vec3<f32> = (camera.viewMat * vec4<f32>(worldN, 0.0)).xyz;

  return FragOut(vec4<f32>(groundColor * lit, 1.0), vec4<f32>(viewN * 0.5 + 0.5, 1.0));
}
