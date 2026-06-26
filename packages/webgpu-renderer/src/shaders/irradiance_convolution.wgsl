// ── Diffuse Irradiance Convolution ──
// Convolves the environment cubemap into a diffuse irradiance map
// by integrating over the hemisphere for each face direction.
//
// Input: environment cubemap (texture_cube<f32>), sampler, MVP uniform
// Output: rgba16float cubemap face

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

  // Tangent-space basis from the normal
  let up: vec3<f32> = vec3<f32>(0.0, 1.0, 0.0);
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

      // Spherical → tangent-space direction
      let sinT: f32 = sin(theta);
      let cosT: f32 = cos(theta);
      let tanSample: vec3<f32> = vec3<f32>(sinT * cos(phi), sinT * sin(phi), cosT);

      // Tangent → world space
      let sampleVec: vec3<f32> = tanSample.x * right + tanSample.y * localUp + tanSample.z * N;

      irradiance += textureSample(envMap, envSampler, sampleVec).rgb * cosT * sinT;
      nrSamples += 1.0;
    }
  }

  irradiance = PI * irradiance / nrSamples;
  return vec4<f32>(irradiance, 1.0);
}
