// ── Debug depth display (separate module to avoid binding conflicts) ──
// Depth textures need texture_depth_2d at @group(0) @binding(0), which
// conflicts with debugTex (texture_2d<f32>) used by fs_debug/fs_copy_color
// in the shared ssao.wgsl module.

@vertex
fn vs_fullscreen(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  let pos = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  return vec4<f32>(pos[i], 0.0, 1.0);
}

@group(0) @binding(0) var debugDepthTex: texture_depth_2d;

@fragment
fn fs_debug_depth(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let d: f32 = textureLoad(debugDepthTex, vec2<i32>(pos.xy), 0);
  return vec4<f32>(d, d, d, 1.0);
}
