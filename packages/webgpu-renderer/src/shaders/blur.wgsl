// ── Bilateral blur (Babylon.js SSAO2 legacy approach) ──
// Reference: references/SSAO/babylon-ssao2.wgsl
//   weight = clamp(1.0 / (0.003 + abs(diff)), 0.0, 30.0)
// No exp() in the inner loop — uses reciprocal (rcp) which is much faster.

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

// Precomputed spatial Gaussian weights for taps 1..5
// exp(-i² * 0.5) for i = 1..5
const SPATIAL_WEIGHTS: array<f32, 5> = array<f32, 5>(
  0.60653066,  // i=1: exp(-0.5)
  0.13533528,  // i=2: exp(-2)
  0.01110900,  // i=3: exp(-4.5)
  0.00033546,  // i=4: exp(-8)
  0.00000373,  // i=5: exp(-12.5)
);

@fragment
fn fs_blur(@builtin(position) pos: vec4<f32>) -> @location(0) f32 {
  let centerDepth: f32 = linearizeDepth(textureLoad(blurDepthTex, vec2<i32>(pos.xy), 0));
  let centerOcc: f32 = textureLoad(blurOcclusionTex, vec2<i32>(pos.xy), 0).r;

  var total: f32 = centerOcc;
  var weightSum: f32 = 1.0;

  // 5 taps on each side = 11 total (Babylon-style, reduced from previous 15)
  // Babylon bilateral weight: clamp(1.0 / (0.003 + abs(diff)), 0.0, 30.0)
  for (var i: i32 = 1; i <= 5; i++) {
    let offset: vec2<i32> = vec2<i32>(blur.dir * f32(i));
    let spatial: f32 = SPATIAL_WEIGHTS[i - 1];

    // +side
    let sx: i32 = clamp(i32(pos.x) + offset.x, 0, i32(blur.screenSize.x) - 1);
    let sy: i32 = clamp(i32(pos.y) + offset.y, 0, i32(blur.screenSize.y) - 1);
    let sd: f32 = linearizeDepth(textureLoad(blurDepthTex, vec2<i32>(sx, sy), 0));
    let diff: f32 = sd - centerDepth;
    let bilateral: f32 = clamp(1.0 / (0.003 + abs(diff)), 0.0, 30.0);
    let w: f32 = bilateral * spatial;
    total += textureLoad(blurOcclusionTex, vec2<i32>(sx, sy), 0).r * w;
    weightSum += w;

    // -side
    let sx2: i32 = clamp(i32(pos.x) - offset.x, 0, i32(blur.screenSize.x) - 1);
    let sy2: i32 = clamp(i32(pos.y) - offset.y, 0, i32(blur.screenSize.y) - 1);
    let sd2: f32 = linearizeDepth(textureLoad(blurDepthTex, vec2<i32>(sx2, sy2), 0));
    let diff2: f32 = sd2 - centerDepth;
    let bilateral2: f32 = clamp(1.0 / (0.003 + abs(diff2)), 0.0, 30.0);
    let w2: f32 = bilateral2 * spatial;
    total += textureLoad(blurOcclusionTex, vec2<i32>(sx2, sy2), 0).r * w2;
    weightSum += w2;
  }

  return total / weightSum;
}
