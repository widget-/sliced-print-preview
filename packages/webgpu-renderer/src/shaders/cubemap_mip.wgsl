// ── Cubemap mip generation ──
// Renders each face at the current mip level by sampling the previous mip level
// with bilinear filtering. Each output pixel is a filtered average of 2x2 input pixels.
//
// NOTE: This shader is NOT currently imported anywhere. The equirect→cubemap pass
// (equirect_to_cubemap.wgsl) renders all 10 mip levels directly from the equirect
// source, so cascade filtering is unnecessary. This file exists as a reference or
// for an alternative mip-generation path.

@group(0) @binding(0) var<uniform> mvp: mat4x4<f32>;
@group(0) @binding(1) var srcCube: texture_cube<f32>;
@group(0) @binding(2) var srcSampler: sampler;
@group(0) @binding(3) var<uniform> srcLevel: f32;

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
  return textureSampleLevel(srcCube, srcSampler, dir, srcLevel);
}
