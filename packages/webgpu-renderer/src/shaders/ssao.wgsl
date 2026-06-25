struct SSAOParams {
  radius: f32,
  intensity: f32,
  bias: f32,
  power: f32,
  _pad: f32,
};

@group(0) @binding(0) var depthTex: texture_2d<f32>;
@group(0) @binding(1) var<uniform> params: SSAOParams;
@group(0) @binding(2) var<uniform> screenSize: vec2<f32>;

@vertex
fn vs_fullscreen(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  let pos = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  return vec4<f32>(pos[i], 0.0, 1.0);
}

// ── SSAO fragment shader ──
// Uses screen-space derivatives (dFdx/dFdy) to estimate normals from depth.
// Samples 12 neighbors in a spiral pattern; checks if each is shallower.
@fragment
fn fs_ssao(@builtin(position) pos: vec4<f32>) -> @location(0) f32 {
  let depth: f32 = textureLoad(depthTex, vec2<i32>(pos.xy), 0).r;

  // Estimate surface normal from depth derivatives (must be in uniform control flow)
  let dzdx: f32 = dpdx(depth);
  let dzdy: f32 = dpdy(depth);

  if (depth >= 1.0) { return 1.0; }
  let normal: vec3<f32> = normalize(vec3<f32>(-dzdx, -dzdy, 1.0));

  let Rpx: f32 = params.radius;
  let bias: f32 = params.bias;
  let nsamp: u32 = 12u;
  var occ: f32 = 0.0;

  for (var s: u32 = 0u; s < nsamp; s++) {
    let a: f32 = f32(s) * 6.28318 / f32(nsamp);
    let r: f32 = (f32(s) + 1.0) / f32(nsamp) * Rpx;
    let sx: i32 = i32(pos.x) + i32(cos(a) * r);
    let sy: i32 = i32(pos.y) + i32(sin(a) * r);
    let cx: i32 = clamp(sx, 0, i32(screenSize.x) - 1);
    let cy: i32 = clamp(sy, 0, i32(screenSize.y) - 1);
    let sd: f32 = textureLoad(depthTex, vec2<i32>(cx, cy), 0).r;

    // Get the 3D vector from center to sample in (screen_x, screen_y, depth) space
    let dx: f32 = f32(sx - i32(pos.x));
    let dy: f32 = f32(sy - i32(pos.y));
    let v: vec3<f32> = vec3<f32>(dx, dy, (sd - depth) * 100.0);

    // Weight by dot with normal (hemisphere check)
    occ += max(0.0, dot(v, normal)) / (dot(v.xy, v.xy) + 0.01);
  }
  occ = 1.0 - 2.0 * params.intensity / f32(nsamp) * occ;
  occ = pow(max(occ, 0.0), params.power);
  return occ;
}

// ── Composite: scene color × occlusion ──
@group(1) @binding(0) var compColorTex: texture_2d<f32>;
@group(1) @binding(1) var compSsaoTex: texture_2d<f32>;

@fragment
fn fs_composite(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let c: vec4<f32> = textureLoad(compColorTex, vec2<i32>(pos.xy), 0);
  let o: vec4<f32> = textureLoad(compSsaoTex, vec2<i32>(pos.xy), 0);
  return vec4<f32>(c.rgb * o.r, c.a);
}

// ── Debug: display a single texture as grayscale ──
@group(0) @binding(0) var debugTex: texture_2d<f32>;

@fragment
fn fs_debug(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let v: vec4<f32> = textureLoad(debugTex, vec2<i32>(pos.xy), 0);
  return vec4<f32>(v.r, v.r, v.r, 1.0);
}

// ── Debug: display a depth texture as grayscale ──
@group(0) @binding(0) var debugDepthTex: texture_depth_2d;

@fragment
fn fs_debug_depth(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let d: f32 = textureLoad(debugDepthTex, vec2<i32>(pos.xy), 0);
  return vec4<f32>(d, d, d, 1.0);
}
