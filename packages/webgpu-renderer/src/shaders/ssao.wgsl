// ── Shared entry points for composite, debug, and copy pipelines ──
// These entry points live in their own module (separate from the SSAO
// fragment shader) to avoid Dawn validation conflicts: they only need
// a subset of the bindings that fs_ssao requires.

@vertex
fn vs_fullscreen(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  let pos = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  return vec4<f32>(pos[i], 0.0, 1.0);
}

// ── Composite: scene color × occlusion × contact shadow ──
@group(1) @binding(0) var compColorTex: texture_2d<f32>;
@group(1) @binding(1) var compSsaoTex: texture_2d<f32>;
@group(1) @binding(2) var compContactShadowTex: texture_2d<f32>;

@fragment
fn fs_composite(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let c: vec4<f32> = textureLoad(compColorTex, vec2<i32>(pos.xy), 0);
  let o: vec4<f32> = textureLoad(compSsaoTex, vec2<i32>(pos.xy), 0);
  let cs: vec4<f32> = textureLoad(compContactShadowTex, vec2<i32>(pos.xy), 0);
  return vec4<f32>(c.rgb * o.r * cs.r, c.a);
}

// ── Debug: display float texture as grayscale (color, occlusion, normal, velocity) ──
@group(0) @binding(0) var debugTex: texture_2d<f32>;

@fragment
fn fs_debug(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let v: vec4<f32> = textureLoad(debugTex, vec2<i32>(pos.xy), 0);
  return vec4<f32>(v.r, v.r, v.r, 1.0);
}

// ── Pass-through copy (SSAO-off path) ──
@fragment
fn fs_copy_color(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  return textureLoad(debugTex, vec2<i32>(pos.xy), 0);
}
