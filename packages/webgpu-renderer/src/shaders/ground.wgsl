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
@group(1) @binding(3) var<uniform> shadowParams: vec4<f32>; // x=softness

// ── group(3): Secondary (fill) light shadow ──
@group(3) @binding(0) var shadowTex2: texture_depth_2d;
@group(3) @binding(1) var shadowSampler2: sampler_comparison;
@group(3) @binding(2) var<uniform> shadowVP2: mat4x4<f32>;
@group(3) @binding(4) var<uniform> shadowParams2: vec4<f32>; // x=softness

// ── Shadow helpers (PCF — rotated Vogel disk + receiver-plane bias) ──

fn interleavedGradientNoise(pos: vec2<f32>) -> f32 {
  return fract(52.9829189 * fract(dot(pos, vec2<f32>(0.06711056, 0.00583715))));
}

fn vogelDiskSample(index: u32, count: u32, phi: f32) -> vec2<f32> {
  let goldenAngle: f32 = 2.399963229728653;
  let r: f32 = sqrt((f32(index) + 0.5) / f32(count));
  let theta: f32 = f32(index) * goldenAngle + phi;
  return vec2<f32>(cos(theta), sin(theta)) * r;
}

fn computeReceiverPlaneDepthBias(p: vec3<f32>) -> vec2<f32> {
  let duvz_dx = dpdx(p);
  let duvz_dy = dpdy(p);
  let inv_det = 1.0 / (duvz_dx.x * duvz_dy.y - duvz_dx.y * duvz_dy.x);
  return vec2<f32>(
    duvz_dy.y * duvz_dx.z - duvz_dx.y * duvz_dy.z,
    duvz_dx.x * duvz_dy.z - duvz_dy.x * duvz_dx.z
  ) * inv_det;
}

fn computeGroundShadow(tex: texture_depth_2d, smp: sampler_comparison, vp: mat4x4<f32>, softness: f32, worldPos: vec3<f32>, clipPos: vec4<f32>) -> f32 {
  let shadowClip: vec4<f32> = vp * vec4<f32>(worldPos, 1.0);
  let shadowNDC: vec3<f32> = shadowClip.xyz / shadowClip.w;
  let shadowUV: vec2<f32> = shadowNDC.xy * vec2<f32>(0.5, -0.5) + 0.5;

  let inFrustum: bool = all(shadowUV == clamp(shadowUV, vec2<f32>(0.0), vec2<f32>(1.0)));
  // No Z bounds check for the ground — the ground plane projects outside the
  // tight shadow frustum's Z range, but the UV clamp + depth comparison still
  // work correctly for the portion that intersects the frustum.

  let dz_duv: vec2<f32> = computeReceiverPlaneDepthBias(shadowNDC);

  var vis: f32 = 1.0;
  if (inFrustum) {
    let texelSize: f32 = 1.0 / 1024.0;
    let radius: f32 = texelSize * softness;
    let phi: f32 = interleavedGradientNoise(clipPos.xy) * 6.283185307;
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < 12u; i++) {
      let offset: vec2<f32> = vogelDiskSample(i, 12u, phi) * radius;
      let uv: vec2<f32> = clamp(shadowUV + offset, vec2<f32>(0.0), vec2<f32>(1.0));
      let perSampleBias: f32 = dot(dz_duv, offset);
      let refZ: f32 = clamp(shadowNDC.z + perSampleBias, 0.0, 1.0);
      sum += textureSampleCompareLevel(tex, smp, uv, refZ);
    }
    vis = sum / 12.0;
  }
  return vis;
}

struct FragOut {
  @location(0) color: vec4<f32>,
  @location(1) normal: vec4<f32>,
};

@fragment
fn fs_ground(in: VSOut) -> FragOut {
  // Ground color (subtle warm gray, like a concrete studio floor)
  let groundColor: vec3<f32> = vec3<f32>(0.35, 0.33, 0.30);

  // Shadow from both lights
  let shadowVis1: f32 = computeGroundShadow(shadowTex, shadowSampler, shadowVP, shadowParams.x, in.worldPos, in.clipPos);
  let shadowVis2: f32 = computeGroundShadow(shadowTex2, shadowSampler2, shadowVP2, shadowParams2.x, in.worldPos, in.clipPos);

  // Combined shadow visibility with ambient fill (both lights)
  let ambientFill: f32 = 0.15;
  let shadowCombined: f32 = min(shadowVis1, shadowVis2);
  let lit: f32 = max(ambientFill, shadowCombined);

  // Write a normal facing +Z (upward) so SSAO at the ground/model boundary is consistent
  let worldN: vec3<f32> = vec3<f32>(0.0, 0.0, 1.0);
  let viewN: vec3<f32> = (camera.viewMat * vec4<f32>(worldN, 0.0)).xyz;

  // DEBUG: uncomment to see shadowNDC.z as grayscale (near=0 black, far=1 white)
  // return FragOut(vec4<f32>(shadowNDC.z, shadowNDC.z, shadowNDC.z, 1.0), vec4<f32>(viewN * 0.5 + 0.5, 1.0));

  return FragOut(vec4<f32>(groundColor * lit, 1.0), vec4<f32>(viewN * 0.5 + 0.5, 1.0));
}
