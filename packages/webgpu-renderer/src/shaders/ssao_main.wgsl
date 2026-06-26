// ── SSAO fragment shader — 3D hemisphere sampling (separate module) ──
// Uses bindings 0-5 in group 0. Declared separately from fs_composite,
// fs_debug etc. so they don't conflict with the shared ssao.wgsl module.

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

@group(0) @binding(0) var depthTex: texture_depth_2d;
@group(0) @binding(1) var<uniform> params: SSAOParams;
@group(0) @binding(2) var<uniform> screenSize: vec2<f32>;
@group(0) @binding(3) var normalTex: texture_2d<f32>;
@group(0) @binding(4) var<storage, read> kernel: array<vec4<f32>, 32>;
@group(0) @binding(5) var<uniform> proj: mat4x4<f32>;

@vertex
fn vs_fullscreen(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  let pos = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  return vec4<f32>(pos[i], 0.0, 1.0);
}

fn linearizeDepth(d: f32) -> f32 {
  return (params.near * params.far) / (d * (params.far - params.near) - params.far);
}

fn viewSpacePos(uv: vec2<f32>, z: f32) -> vec3<f32> {
  let f: f32 = 2.0 / params.fovScale;
  let aspect: f32 = screenSize.x / screenSize.y;
  let ndc: vec2<f32> = vec2<f32>(uv.x * 2.0 - 1.0, -(uv.y * 2.0 - 1.0));
  let absZ: f32 = -z;
  return vec3<f32>(ndc.x * absZ * aspect / f, ndc.y * absZ / f, z);
}

fn hash3(p: vec2<u32>) -> vec3<f32> {
  let ph: u32 = p.x * 374761393u + p.y * 668265263u;
  let ph2: u32 = ph ^ (ph >> 13u);
  let ph3: u32 = ph2 * 1274126177u;
  let x: f32 = f32(ph3 & 0xFFFFu) / 32767.0 - 1.0;
  let y: f32 = f32((ph3 >> 8u) & 0xFFFFu) / 32767.0 - 1.0;
  let z: f32 = f32((ph3 >> 16u) & 0xFFFFu) / 32767.0 - 1.0;
  return normalize(vec3<f32>(x, y, z));
}

@fragment
fn fs_ssao(@builtin(position) pos: vec4<f32>) -> @location(0) f32 {
  let depth: f32 = textureLoad(depthTex, vec2<i32>(pos.xy), 0);
  if (depth >= 1.0) { return 1.0; }

  let linDepth: f32 = linearizeDepth(depth);
  let uv: vec2<f32> = pos.xy / screenSize;
  let normalEnc: vec4<f32> = textureLoad(normalTex, vec2<i32>(pos.xy), 0);
  let normal: vec3<f32> = normalize(normalEnc.xyz * 2.0 - 1.0);
  let viewPos: vec3<f32> = viewSpacePos(uv, linDepth);

  let randomVec: vec3<f32> = hash3(vec2<u32>(u32(pos.x), u32(pos.y)));
  let tangent: vec3<f32> = normalize(randomVec - normal * dot(randomVec, normal));
  let bitangent: vec3<f32> = cross(normal, tangent);
  let TBN: mat3x3<f32> = mat3x3<f32>(tangent, bitangent, normal);

  var occ: f32 = 0.0;
  for (var i: u32 = 0u; i < 32u; i++) {
    let sampleDir: vec3<f32> = TBN * kernel[i].xyz;
    let samplePos: vec3<f32> = viewPos + sampleDir * params.radius;

    let clipPos: vec4<f32> = proj * vec4<f32>(samplePos, 1.0);
    let ndc: vec2<f32> = clipPos.xy / clipPos.w;
    let sampleUV: vec2<f32> = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);

    if (sampleUV.x < 0.0 || sampleUV.x > 1.0 || sampleUV.y < 0.0 || sampleUV.y > 1.0) { continue; }

    let sampCoord: vec2<i32> = vec2<i32>(sampleUV * screenSize);
    let sampleDepth: f32 = textureLoad(depthTex, sampCoord, 0);
    if (sampleDepth >= 1.0) { continue; }

    let sampleLin: f32 = linearizeDepth(sampleDepth);
    let depthDiff: f32 = sampleLin - linDepth;
    let rangeCheck: f32 = smoothstep(0.0, params.radius, abs(depthDiff));

    if (depthDiff > params.bias) {
      occ += 1.0 * rangeCheck;
    }
  }

  occ = 1.0 - params.intensity * (occ / 32.0);
  occ = pow(max(occ, 0.0), params.power);
  return occ;
}
