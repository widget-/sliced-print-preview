// ── Diffuse Irradiance Convolution ──
//
// Purpose:
//   Convolves the environment cubemap into a low-resolution (32×32 per face)
//   diffuse irradiance cubemap. Each texel integrates the hemisphere above
//   its face normal using uniform spherical sampling.
//
// Input:
//   envMap: texture_cube<f32> — the RGBE-loaded cubemap (512×512 per face, 10 mips, rgba16float)
//   envSampler: sampler       — linear filtering for envMap samples
//   mvp: mat4x4<f32>          — face-specific view-projection matrix
//
// Output:
//   rgba16float cubemap (32×32 per face, single mip level)
//   Written to binding 0 of the IBL group (irradianceMap) in pipeline.ts
//
// Algorithm:
//   For each face direction N, integrate over the hemisphere:
//     E(N) = (1/π) ∫_Ω L(ω) cos(θ) dω
//   where L(ω) is the incoming radiance from the env cubemap, sampled at
//   direction ω = tanSample in world space.
//
//   Integration uses a fixed-step spherical grid:
//     sampleDelta = 0.025 radians (~1.43°)
//     phi steps:   ⌈2π / ∆⌉ ≈ 252
//     theta steps: ⌈π/2 / ∆⌉ ≈ 63
//     Total: ~15,876 samples per face direction
//
//   Each sample: irradiance += textureSample(envMap, ω).rgb × cosθ × sinθ
//   The sinθ factor accounts for the spherical coordinate area element.
//
// Coordinate system:
//   The tangent frame is built from the face normal N using Z-up convention
//   (world Z = vertical axis). The hemisphere is sampled in tangent space
//   where Z = N (the normal). Samples are transformed to world space via
//   the orthonormal basis (right, localUp, N).
//
//   Robust against N ≈ Z: when |N.z| > 0.999, falls back to +X as the
//   "up" axis for basis construction (avoids degenerate cross product).
//
// Precision:
//   Input:  rgba16float env cubemap (512×512, 10 mips)
//   Output: rgba16float irradiance cubemap (32×32, 1 mip)
//   16-bit float provides sufficient precision for the low-frequency
//   diffuse irradiance signal.

@group(0) @binding(0) var<uniform> mvp: mat4x4<f32>;
@group(0) @binding(1) var envMap: texture_cube<f32>;
@group(0) @binding(2) var envSampler: sampler;

const PI: f32 = 3.14159265359;

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
};

@vertex
fn vs_main(@location(0) position: vec4<f32>) -> VSOut {
  var out: VSOut;
  out.pos = mvp * position;
  out.worldPos = position.xyz;
  return out;
}

@fragment
fn fs_main(@location(0) worldPos: vec3<f32>) -> @location(0) vec4<f32> {
  let N: vec3<f32> = normalize(worldPos);

  // Tangent-space basis from the normal (robust against N ≈ Z-up)
  // Default: world Z = up. When N is nearly parallel to Z (|N.z| ≈ 1),
  // cross(up, N) would be degenerate, so fall back to +X as up.
  let up: vec3<f32> = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(N.z) < 0.999);
  let right: vec3<f32> = normalize(cross(up, N));
  let localUp: vec3<f32> = cross(N, right);

  var irradiance: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
  var nrSamples: f32 = 0.0;

  let sampleDelta: f32 = 0.025;
  let stepsPhi: u32 = u32(ceil(2.0 * PI / sampleDelta));
  let stepsTheta: u32 = u32(ceil(PI * 0.5 / sampleDelta));

  for (var ip: u32 = 0u; ip < stepsPhi; ip++) {
    for (var it: u32 = 0u; it < stepsTheta; it++) {
      let phi: f32 = (f32(ip) + 0.5) * 2.0 * PI / f32(stepsPhi);
      let theta: f32 = (f32(it) + 0.5) * PI * 0.5 / f32(stepsTheta);

      // Spherical → tangent-space direction (z = up = N)
      let sinT: f32 = sin(theta);
      let cosT: f32 = cos(theta);
      let tanSample: vec3<f32> = vec3<f32>(sinT * cos(phi), sinT * sin(phi), cosT);

      // Tangent → world space
      let sampleVec: vec3<f32> = tanSample.x * right + tanSample.y * localUp + tanSample.z * N;

      // Accumulate: L(ω) × cos(θ) × sin(θ)   (sinθ = spherical area element)
      irradiance += textureSample(envMap, envSampler, sampleVec).rgb * cosT * sinT;
      nrSamples += 1.0;
    }
  }

  // Normalize: divide by sample count, multiply by π (Lambertian diffuse integral)
  irradiance = PI * irradiance / nrSamples;
  return vec4<f32>(irradiance, 1.0);
}
