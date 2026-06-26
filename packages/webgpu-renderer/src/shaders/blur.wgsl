// ── Bilateral blur (separate module to avoid binding conflicts with SSAO) ──

struct SSAOParams {
  radius: f32,
  intensity: f32,
  bias: f32,
  power: f32,
  near: f32,
  far: f32,
  fovScale: f32,
  _pad: f32,
};

struct BlurParams {
  dir: vec2<f32>,
  screenSize: vec2<f32>,
};

@group(0) @binding(0) var blurOcclusionTex: texture_2d<f32>;
@group(0) @binding(1) var blurDepthTex: texture_depth_2d;
@group(0) @binding(2) var<uniform> blur: BlurParams;
@group(0) @binding(3) var<uniform> nearFar: vec2<f32>;

fn linearizeDepth(d: f32) -> f32 {
  return (nearFar.x * nearFar.y) / (d * (nearFar.y - nearFar.x) - nearFar.y);
}

@vertex
fn vs_fullscreen(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  let pos = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  return vec4<f32>(pos[i], 0.0, 1.0);
}

@fragment
fn fs_blur(@builtin(position) pos: vec4<f32>) -> @location(0) f32 {
  let centerDepth: f32 = linearizeDepth(textureLoad(blurDepthTex, vec2<i32>(pos.xy), 0));
  let centerOcc: f32 = textureLoad(blurOcclusionTex, vec2<i32>(pos.xy), 0).r;

  var total: f32 = centerOcc;
  var weightSum: f32 = 1.0;

  // 7 taps on each side = 15 total
  for (var i: i32 = 1; i <= 7; i++) {
    let offset: vec2<i32> = vec2<i32>(blur.dir * f32(i));
    let spatial: f32 = exp(-f32(i * i) * 0.5);

    // +side
    let sx: i32 = clamp(i32(pos.x) + offset.x, 0, i32(blur.screenSize.x) - 1);
    let sy: i32 = clamp(i32(pos.y) + offset.y, 0, i32(blur.screenSize.y) - 1);
    let sd: f32 = linearizeDepth(textureLoad(blurDepthTex, vec2<i32>(sx, sy), 0));
    let bilateral: f32 = exp(-((sd - centerDepth) * (sd - centerDepth)) * 20.0);
    let w: f32 = bilateral * spatial;
    total += textureLoad(blurOcclusionTex, vec2<i32>(sx, sy), 0).r * w;
    weightSum += w;

    // -side
    let sx2: i32 = clamp(i32(pos.x) - offset.x, 0, i32(blur.screenSize.x) - 1);
    let sy2: i32 = clamp(i32(pos.y) - offset.y, 0, i32(blur.screenSize.y) - 1);
    let sd2: f32 = linearizeDepth(textureLoad(blurDepthTex, vec2<i32>(sx2, sy2), 0));
    let bilateral2: f32 = exp(-((sd2 - centerDepth) * (sd2 - centerDepth)) * 20.0);
    let w2: f32 = bilateral2 * spatial;
    total += textureLoad(blurOcclusionTex, vec2<i32>(sx2, sy2), 0).r * w2;
    weightSum += w2;
  }

  return total / weightSum;
}
