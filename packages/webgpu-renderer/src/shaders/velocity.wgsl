// ── Velocity buffer: compute per-pixel screen-space motion from depth ──

struct VelocityParams {
  invViewProj: mat4x4<f32>,
  prevViewProj: mat4x4<f32>,
};

@group(0) @binding(0) var depthTex: texture_2d<f32>;
@group(0) @binding(1) var<uniform> params: VelocityParams;

@vertex
fn vs_fullscreen(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  let pos = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  return vec4<f32>(pos[i], 0.0, 1.0);
}

@fragment
fn fs_velocity(@builtin(position) pos: vec4<f32>) -> @location(0) vec2<f32> {
  let depth: f32 = textureLoad(depthTex, vec2<i32>(pos.xy), 0).r;
  let screenSize: vec2<f32> = vec2<f32>(f32(textureDimensions(depthTex, 0).x), f32(textureDimensions(depthTex, 0).y));

  // Current frame NDC (non-linear depth)
  let uv: vec2<f32> = pos.xy / screenSize;
  let ndc: vec4<f32> = vec4<f32>(uv.x * 2.0 - 1.0, -(uv.y * 2.0 - 1.0), depth, 1.0);

  // World-space position via inverse view-projection
  let world: vec4<f32> = params.invViewProj * ndc;
  let worldPos: vec3<f32> = world.xyz / world.w;

  // Previous frame clip space
  let prevClip: vec4<f32> = params.prevViewProj * vec4<f32>(worldPos, 1.0);
  let prevNDC: vec2<f32> = prevClip.xy / prevClip.w;

  // Screen-space velocity = prev_NDC - current_NDC
  return prevNDC - ndc.xy;
}
