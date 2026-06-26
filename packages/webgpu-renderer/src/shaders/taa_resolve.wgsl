// ── TAA resolve: blend current frame with reprojected history ──
// References:
//   Filament (Google) — variance AABB, tonemap blend
//   Godot — disocclusion detection
//   Playdead — original AABB clipping, bilinear history sample

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

// Tonemap operators (Reinhard) for blending in perceptual space
// (ref: Filament — reduces ghosting on HDR content)
fn tonemap(c: vec3<f32>) -> vec3<f32> { return c / (c + 1.0); }
fn invTonemap(c: vec3<f32>) -> vec3<f32> { return c / max(1.0 - c, vec3<f32>(0.001)); }

// Manual bilinear texture sample (avoids needing a sampler in bind group)
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

// Clip history colour to AABB to prevent ghosting
// Playdead-style: clamp the difference vector q-p to the AABB extent from p.
// (ref: Mikkel Gjøl, Morten S. Mikkelsen — "Playdead's TAA")
fn clipAABB(aabbMin: vec3<f32>, aabbMax: vec3<f32>, p: vec3<f32>, q: vec3<f32>) -> vec3<f32> {
  let r: vec3<f32> = q - p;
  let clamped: vec3<f32> = min(max(r, aabbMin - p), aabbMax - p);
  return p + clamped;
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

  // Current frame color (composite: scene × occlusion)
  let currentColor: vec4<f32> = textureLoad(colorTex, icoord, 0);

  // Velocity (in NDC [-1,1] space)
  let velocity: vec2<f32> = textureLoad(velocityTex, icoord, 0).rg;

  // Reproject: history UV = current UV + velocity/2
  // (velocity is in NDC [-1,1] Y-up, divide by 2 to get UV offset.
  //  NDC Y is inverted relative to UV Y, so negate it.)
  let historyUV: vec2<f32> = uv + velocity * vec2<f32>(.5, -.5);
  // Sample history with manual bilinear
  var historyColor: vec3<f32> = sampleBilinear(historyTex, historyUV, screenSize);

  // Disocclusion detection: compare velocity at reprojected UV with current velocity.
  // If they differ significantly, a different object is now at the reprojected location
  // (background revealed by camera motion). Reduce history contribution.
  // (ref: Godot — disocclusion detection via velocity comparison)
  let historyCoord: vec2<i32> = vec2<i32>(clamp(historyUV * screenSize, vec2<f32>(0.0), screenSize - 1.0));
  let historyVelocity: vec2<f32> = textureLoad(velocityTex, historyCoord, 0).rg;
  let velDiff: f32 = length(historyVelocity - velocity);
  let disocclusion: f32 = smoothstep(0.001, 0.02, velDiff);

  // Velocity-adaptive blend: mostly history for moire suppression, but enough
  // current frame (10%) so the image converges to sharp over ~10 frames.
  // High-frequency detail gets a gentle extra nudge toward history.
  let motionBlend: f32 = smoothstep(0.001, 0.01, length(velocity));
  let adaptiveBlend: f32 = mix(0.15, params.blendFactor, motionBlend);

  // 3×3 neighborhood for AABB clipping
  let offsets: array<vec2<i32>, 8> = array<vec2<i32>, 8>(
    vec2<i32>(-1, -1), vec2<i32>(0, -1), vec2<i32>(1, -1),
    vec2<i32>(-1,  0),                    vec2<i32>(1,  0),
    vec2<i32>(-1,  1), vec2<i32>(0,  1), vec2<i32>(1,  1),
  );
  var cMin: vec3<f32> = currentColor.rgb;
  var cMax: vec3<f32> = currentColor.rgb;
  var cMean: vec3<f32> = currentColor.rgb;
  for (var i: i32 = 0; i < 8; i++) {
    let s: vec2<i32> = clamp(icoord + offsets[i], vec2<i32>(0, 0), vec2<i32>(screenSize - 1));
    let c: vec3<f32> = textureLoad(colorTex, s, 0).rgb;
    cMin = min(cMin, c);
    cMax = max(cMax, c);
    cMean += c;
  }
  cMean = (cMean + currentColor.rgb) / 9.0;

  // Variance-based AABB expansion: widen the box slightly based on local variance
  // to reduce flickering while still preventing ghosting (ref: Filament)
  let aabbCenter: vec3<f32> = (cMin + cMax) * 0.5;
  let varianceExpand: vec3<f32> = abs(cMean - aabbCenter) * 0.5;
  cMin = cMin - varianceExpand;
  cMax = cMax + varianceExpand;

  // Moire suppression: high-frequency detail (layer lines) gets a gentle extra
  // nudge toward history. Reduces blend by at most 40% on the strongest edges.
  // This is subtle — the 90% history base does the heavy lifting.
  let localRange: f32 = max(max(cMax.r - cMin.r, cMax.g - cMin.g), cMax.b - cMin.b);
  let moireFactor: f32 = 1.0 - 0.4 * smoothstep(0.05, 0.3, localRange);
  let effectiveBlend: f32 = max(adaptiveBlend * moireFactor, disocclusion);

  // Tonemap before clipping for better HDR handling (ref: Filament, Godot)
  let currentTone: vec3<f32> = tonemap(currentColor.rgb);
  let historyTone: vec3<f32> = tonemap(historyColor);
  let clipMinTone: vec3<f32> = tonemap(cMin);
  let clipMaxTone: vec3<f32> = tonemap(cMax);

  // Clip history to neighborhood AABB (in tonemapped space)
  let clampedTone: vec3<f32> = clipAABB(clipMinTone, clipMaxTone, currentTone, historyTone);

  // Blend: lerp(clamped_history, current, blendFactor) in tonemapped space
  let resultTone: vec3<f32> = mix(clampedTone, currentTone, vec3<f32>(effectiveBlend));

  // Un-tonemap
  var outColor: vec3<f32> = invTonemap(resultTone);

  // Disocclusion: if reprojected UV is out of screen, use current frame
  if (historyUV.x < 0.0 || historyUV.x >= 1.0 || historyUV.y < 0.0 || historyUV.y >= 1.0) {
    outColor = currentColor.rgb;
  }

  let finalColor: vec4<f32> = vec4<f32>(outColor, 1.0);
  return TAAOutput(finalColor, finalColor);
}
