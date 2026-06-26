// ── Screen-space contact shadows ──
// Ray-marches from each surface point toward the light direction in screen
// space using the depth buffer. Captures fine occlusion (layer gaps, nearby
// geometry) that the shadow map misses.
//
// Reference: shadow-filament.md §Screen-Space Contact Shadows
//            references/contact-shadows/README.md
//
// Algorithm:
//   1. Reconstruct world position from depth + invViewProj
//   2. Project world-space ray (pos, pos + lightDir * maxDist) into UV space
//   3. March along UV ray, comparing ray depth against stored depth
//   4. If ray passes behind a surface within thickness → occluded
//   5. Apply screen-edge fade
//
// Uses textureLoad (integer coords) instead of textureSampleLevel because
// depth textures cannot be sampled with a filtering sampler. This means
// nearest-neighbor depth reads — acceptable with IGN dithering.

struct ContactShadowParams {
  invViewProj: mat4x4<f32>,   // world pos reconstruction
  viewProj: mat4x4<f32>,      // world→clip for ray projection
  lightDir: vec4<f32>,        // xyz=normalized direction, w=strength
  params: vec4<f32>,          // x=maxDist, y=stepCount, z=linearThickness, w=edgeFadeDist
};

@group(0) @binding(0) var depthTex: texture_depth_2d;
@group(0) @binding(1) var<uniform> cs: ContactShadowParams;

fn reconstructWorldPos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
  // uv is Y-down (from @builtin(position)), NDC is Y-up — flip Y to match invViewProj
  let ndc: vec4<f32> = vec4<f32>(uv.x * 2.0 - 1.0, -(uv.y * 2.0 - 1.0), depth, 1.0);
  let world: vec4<f32> = cs.invViewProj * ndc;
  return world.xyz / world.w;
}

fn interleavedGradientNoise(pos: vec2<f32>) -> f32 {
  return fract(52.9829189 * fract(dot(pos, vec2<f32>(0.06711056, 0.00583715))));
}

@vertex
fn vs_fullscreen(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  let pos = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  return vec4<f32>(pos[i], 0.0, 1.0);
}

@fragment
fn fs_contact_shadow(@builtin(position) pos: vec4<f32>) -> @location(0) f32 {
  let screenSize: vec2<u32> = textureDimensions(depthTex);
  let screenSizeF: vec2<f32> = vec2<f32>(screenSize);
  let uv: vec2<f32> = pos.xy / screenSizeF;

  let depth: f32 = textureLoad(depthTex, vec2<i32>(pos.xy), 0i);

  // Background (far plane) — no shadow to cast
  if (depth >= 1.0) { return 1.0; }

  let worldPos: vec3<f32> = reconstructWorldPos(uv, depth);
  let lightDir: vec3<f32> = normalize(cs.lightDir.xyz);
  let maxDist: f32 = cs.params.x;
  let stepCount: i32 = i32(cs.params.y);
  let edgeFadeDist: f32 = cs.params.w;

  // Ray start/end in world space, projected to clip space
  let rayEnd: vec3<f32> = worldPos + lightDir * maxDist;
  let csStart: vec4<f32> = cs.viewProj * vec4<f32>(worldPos, 1.0);
  let csEnd: vec4<f32>   = cs.viewProj * vec4<f32>(rayEnd, 1.0);

  // Clip → NDC → UV (with Y flip: WebGPU NDC Y-up → texture UV Y-down)
  let ndcStart: vec3<f32> = csStart.xyz / csStart.w;
  let ndcEnd: vec3<f32>   = csEnd.xyz   / csEnd.w;
  let uvStart: vec3<f32> = vec3<f32>(ndcStart.x * 0.5 + 0.5, ndcStart.y * -0.5 + 0.5, ndcStart.z);
  let uvEnd: vec3<f32>   = vec3<f32>(ndcEnd.x   * 0.5 + 0.5, ndcEnd.y   * -0.5 + 0.5, ndcEnd.z);

  // Screen-space ray length → adaptive step count
  let ssLen: f32 = length(uvEnd.xy - uvStart.xy);
  let steps: i32 = min(max(i32(ssLen * 128.0), 1), stepCount);
  let step: vec3<f32> = (uvEnd - uvStart) / vec3<f32>(f32(steps));

  // Dither the start offset to reduce banding
  let dither: f32 = interleavedGradientNoise(pos.xy) * 0.5 + 0.5;
  var sampleUV: vec3<f32> = uvStart + step * dither;

  // Adaptive depth tolerance: per-step Z change along the ray.
  // Capped at 0.01 NDC Z to prevent over-occlusion on steep surfaces
  // where the ray spans a large Z range (making dzTolerance too permissive).
  // (ref: Filament — adaptive, with empirical cap)
  let dzTolerance: f32 = min(abs(uvEnd.z - uvStart.z) / f32(steps), 0.01);

  var occlusion: f32 = 1.0;
  for (var i: i32 = 0; i < steps; i++) {
    sampleUV += step;

    // Clamp UV to screen bounds (skip pixels outside viewport)
    if (any(sampleUV.xy < vec2<f32>(0.0)) || any(sampleUV.xy > vec2<f32>(1.0))) { break; }

    let smpPos: vec2<i32> = vec2<i32>(sampleUV.xy * screenSizeF);
    let sampleDepth: f32 = textureLoad(depthTex, smpPos, 0i);
    let diff: f32 = sampleUV.z - sampleDepth;

    // Filament-style occlusion test: ray is within tolerance of the surface
    // dzTolerance is the per-step NDC Z change — surfaces within this range
    // are close enough to block the light (contact shadow).
    // This naturally handles perspective compression: the tolerance is smaller
    // near the camera (where NDC Z changes rapidly) and larger at distance.
    if (diff > 0.0 && diff < dzTolerance) {
      occlusion = f32(i) / f32(steps);
      break;
    }
  }

  // Screen-edge fade
  let edgeFade: f32 = 1.0 - saturate(dot(
    max(vec2<f32>(0.0), abs(uv.xy * 2.0 - 1.0) - (1.0 - edgeFadeDist)) / edgeFadeDist,
    vec2<f32>(1.0)
  ));
  occlusion = mix(1.0, occlusion, edgeFade);

  // Apply contact shadow strength (lightDir.w = strength multiplier)
  occlusion = mix(1.0, occlusion, cs.lightDir.w);

  return occlusion;
}
