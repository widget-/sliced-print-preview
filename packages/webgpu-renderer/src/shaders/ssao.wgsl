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

@group(0) @binding(0) var depthTex: texture_2d<f32>;
@group(0) @binding(1) var<uniform> params: SSAOParams;
@group(0) @binding(2) var<uniform> screenSize: vec2<f32>;

@vertex
fn vs_fullscreen(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  let pos = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  return vec4<f32>(pos[i], 0.0, 1.0);
}

fn linearizeDepth(d: f32) -> f32 {
  // Returns view-space Z (negative for points in front of camera).
  // Projection: proj[10] = -(far+near)/(far-near), proj[14] = -2*far*near/(far-near)
  // depth = 0.5 * (proj[10] * viewZ + proj[14]) / (-viewZ) + 0.5
  // → viewZ = far*near / (depth*(far-near) - far)
  return (params.near * params.far) / (d * (params.far - params.near) - params.far);
}

fn viewSpacePos(uv: vec2<f32>, z: f32) -> vec3<f32> {
  // Reconstruct view-space XYZ from NDC coords and view-space Z.
  // f = 1/tan(fov/2), fovScale = 2*tan(fov/2) → f = 2/fovScale
  let f: f32 = 2.0 / params.fovScale;
  let aspect: f32 = screenSize.x / screenSize.y;
  let ndc: vec2<f32> = vec2<f32>(uv.x * 2.0 - 1.0, -(uv.y * 2.0 - 1.0));
  let absZ: f32 = -z; // positive distance from camera
  return vec3<f32>(ndc.x * absZ * aspect / f, ndc.y * absZ / f, z);
}

// ── SSAO fragment shader ──
// Reconstructs view-space positions and uses proper geometric normals.
@fragment
fn fs_ssao(@builtin(position) pos: vec4<f32>) -> @location(0) f32 {
  let depth: f32 = textureLoad(depthTex, vec2<i32>(pos.xy), 0).r;
  if (depth >= 1.0) { return 1.0; }

  let linDepth: f32 = linearizeDepth(depth); // view-space Z, negative

  // Reconstruct view-space position and compute geometric normal
  let uv: vec2<f32> = pos.xy / screenSize;
  let viewPos: vec3<f32> = viewSpacePos(uv, linDepth);
  let vpDx: vec3<f32> = dpdx(viewPos);
  let vpDy: vec3<f32> = dpdy(viewPos);
  // cross(ddx,ddy) points into the surface (–Z). Flip it so the
  // hemisphere test checks samples IN FRONT (toward camera, +Z).
  let normal: vec3<f32> = normalize(-cross(vpDx, vpDy));

  let Rpx: f32 = params.radius;
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
    if (sd >= 1.0) { continue; }

    let sLinDepth: f32 = linearizeDepth(sd);
    let sampleUV: vec2<f32> = vec2<f32>(f32(cx), f32(cy)) / screenSize;
    let samplePos: vec3<f32> = viewSpacePos(sampleUV, sLinDepth);

    // 3D vector from center to sample in view space
    let v: vec3<f32> = samplePos - viewPos;
    let depthDiff: f32 = sLinDepth - linDepth;

    // Self-occlusion prevention (reject samples in front of the surface)
    if (depthDiff < -params.bias) { continue; }

    // Smooth depth range falloff: samples beyond the occlusion radius
    // contribute progressively less, avoiding hard-cutoff artifacts.
    let maxDist: f32 = Rpx * (-linDepth) * params.fovScale / screenSize.y;
    let rangeWeight: f32 = 1.0 - smoothstep(maxDist * 0.3, maxDist, depthDiff);

    // Smooth distance falloff: closer samples contribute more.
    let distSq: f32 = dot(v, v);
    let distWeight: f32 = 1.0 / (distSq + 0.01);

    // Hemisphere check × smooth falloffs
    occ += max(0.0, dot(v, normal)) * rangeWeight * distWeight;
  }

  occ = 1.0 - params.intensity * 2.0 / f32(nsamp) * occ;
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
