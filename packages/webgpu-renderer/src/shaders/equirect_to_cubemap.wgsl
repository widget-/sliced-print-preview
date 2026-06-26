// ── Equirectangular → Cubemap ──
// Renders each cubemap face by sampling the equirectangular HDR.
// Input: equirectHDR (texture_2d<f32>), sampler, MVP uniform
// Output: rgba16float cubemap face

@group(0) @binding(0) var<uniform> mvp: mat4x4<f32>;
@group(0) @binding(1) var equirectTex: texture_2d<f32>;
@group(0) @binding(2) var equirectSampler: sampler;

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
  let dir = normalize(worldPos);
  // Spherical coords from direction
  let phi = atan2(dir.z, dir.x);
  let theta = asin(dir.y);
  let uv = vec2<f32>(phi * 0.1591549 + 0.5, theta * 0.3183099 + 0.5);
  let color = textureSample(equirectTex, equirectSampler, uv).rgb;
  return vec4<f32>(color, 1.0);
}
