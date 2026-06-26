// ── TAA resolve: blend current frame with reprojected history ──

struct TAAParams {
  blendFactor: f32,
  _pad0: f32,
  _pad1: f32,
  _pad2: f32,
};

@group(0) @binding(0) var colorTex: texture_2d<f32>;
@group(0) @binding(1) var velocityTex: texture_2d<f32>;
@group(0) @binding(2) var historyTex: texture_2d<f32>;

@group(0) @binding(3) var<uniform> params: TAAParams;

@vertex
fn vs_fullscreen(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  let pos = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  return vec4<f32>(pos[i], 0.0, 1.0);
}

// Clip history colour to AABB to prevent ghosting
// Based on Playdead's TAA (Mikkel Gjøl, Morten S. Mikkelsen)
fn clipAABB(aabbMin: vec3<f32>, aabbMax: vec3<f32>, p: vec3<f32>, q: vec3<f32>) -> vec3<f32> {
  let r: vec3<f32> = q - p;
  var t: f32 = 1.0;
  if (r.x != 0.0) {
    t = min(t, max((aabbMin.x - p.x) / r.x, 0.0));
    t = min(t, max((aabbMax.x - p.x) / r.x, 0.0));
  }
  if (r.y != 0.0) {
    t = min(t, max((aabbMin.y - p.y) / r.y, 0.0));
    t = min(t, max((aabbMax.y - p.y) / r.y, 0.0));
  }
  if (r.z != 0.0) {
    t = min(t, max((aabbMin.z - p.z) / r.z, 0.0));
    t = min(t, max((aabbMax.z - p.z) / r.z, 0.0));
  }
  return p + r * t;
}

// Manual bilinear sample (avoids needing a sampler in the bind group)
fn sampleBilinear(tex: texture_2d<f32>, uv: vec2<f32>, screenSize: vec2<f32>) -> vec3<f32> {
  let p: vec2<f32> = uv * screenSize - 0.5;
  let xy: vec2<i32> = vec2<i32>(max(p, vec2<f32>(0.0)));
  let f: vec2<f32> = p - vec2<f32>(xy);
  let c00: vec3<f32> = textureLoad(tex, xy + vec2<i32>(0, 0), 0).rgb;
  let c10: vec3<f32> = textureLoad(tex, min(xy + vec2<i32>(1, 0), vec2<i32>(screenSize - 1)), 0).rgb;
  let c01: vec3<f32> = textureLoad(tex, min(xy + vec2<i32>(0, 1), vec2<i32>(screenSize - 1)), 0).rgb;
  let c11: vec3<f32> = textureLoad(tex, min(xy + vec2<i32>(1, 1), vec2<i32>(screenSize - 1)), 0).rgb;
  return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}

struct TAAOutput {
  @location(0) color: vec4<f32>,
  @location(1) history: vec4<f32>,
};

@fragment
fn fs_taa(@builtin(position) pos: vec4<f32>) -> TAAOutput {
  let screenSize: vec2<f32> = vec2<f32>(f32(textureDimensions(colorTex, 0).x), f32(textureDimensions(colorTex, 0).y));
  let uv: vec2<f32> = pos.xy / screenSize;
  let icoord: vec2<i32> = vec2<i32>(pos.xy);

  // Current frame color
  let currentColor: vec4<f32> = textureLoad(colorTex, icoord, 0);

  // Velocity (in NDC [-1,1] space)
  let velocity: vec2<f32> = textureLoad(velocityTex, icoord, 0).rg;

  // Reproject: history UV = current UV + velocity/2
  let historyUV: vec2<f32> = uv + velocity * 0.5;

  // Sample history with manual bilinear
  var historyColor: vec3<f32> = sampleBilinear(historyTex, historyUV, screenSize);

  // Neighborhood AABB from 3×3 current frame (for clamping history)
  let offsets: array<vec2<i32>, 8> = array<vec2<i32>, 8>(
    vec2<i32>(-1, -1), vec2<i32>(0, -1), vec2<i32>(1, -1),
    vec2<i32>(-1,  0),                    vec2<i32>(1,  0),
    vec2<i32>(-1,  1), vec2<i32>(0,  1), vec2<i32>(1,  1),
  );
  var cMin: vec3<f32> = currentColor.rgb;
  var cMax: vec3<f32> = currentColor.rgb;
  for (var i: i32 = 0; i < 8; i++) {
    let s: vec2<i32> = clamp(icoord + offsets[i], vec2<i32>(0, 0), vec2<i32>(screenSize - 1));
    let c: vec3<f32> = textureLoad(colorTex, s, 0).rgb;
    cMin = min(cMin, c);
    cMax = max(cMax, c);
  }

  // Clip history to neighborhood AABB (anti-ghosting)
  let clamped: vec3<f32> = clipAABB(cMin, cMax, currentColor.rgb, historyColor);

  // Blend: lerp(clamped_history, current, blendFactor)
  let result: vec3<f32> = mix(clamped, currentColor.rgb, vec3<f32>(params.blendFactor));

  // Disocclusion: if reprojected UV is out of screen, use current frame
  var outColor: vec3<f32> = result;
  if (historyUV.x < 0.0 || historyUV.x >= 1.0 || historyUV.y < 0.0 || historyUV.y >= 1.0) {
    outColor = currentColor.rgb;
  }

  let finalColor: vec4<f32> = vec4<f32>(outColor, 1.0);
  return TAAOutput(finalColor, finalColor);
}
