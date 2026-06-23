struct SSAOParams {
  radius: f32,
  intensity: f32,
  bias: f32,
  power: f32,
  _pad: f32,
};

struct ProjParams {
  p11: f32,
  p22: f32,
  near: f32,
  far: f32,
};

@group(0) @binding(0) var depthTex: texture_2d<f32>;
@group(0) @binding(1) var normalTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var ssaoOut: texture_storage_2d<r8unorm, write>;
@group(0) @binding(3) var<uniform> params: SSAOParams;
@group(0) @binding(4) var<uniform> proj: ProjParams;

// ── Normal reconstruction from depth ──
@compute @workgroup_size(16, 16)
fn cs_normal(@builtin(global_invocation_id) id: vec3<u32>) {
  let size: vec2<u32> = textureDimensions(depthTex);
  if (id.x >= size.x || id.y >= size.y) { return; }

  let uv: vec2<f32> = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(size);
  let depth: f32 = textureLoad(depthTex, vec2<i32>(id.xy), 0).r;
  if (depth >= 1.0) { textureStore(normalTex, vec2<i32>(id.xy), vec4<f32>(0.0, 0.0, 1.0, 1.0)); return; }

  let texel: vec2<f32> = 1.0 / vec2<f32>(size);
  let p11: f32 = proj.p11;
  let p22: f32 = proj.p22;

  let ndc: vec2<f32> = uv * 2.0 - 1.0;
  let vPos: vec3<f32> = vec3<f32>(ndc * depth / vec2<f32>(p11, p22), depth);

  let uvL: vec2<f32> = (vec2<f32>(id.xy + vec2<i32>(-1, 0)) + 0.5) / vec2<f32>(size);
  let uvR: vec2<f32> = (vec2<f32>(id.xy + vec2<i32>(1, 0)) + 0.5) / vec2<f32>(size);
  let uvU: vec2<f32> = (vec2<f32>(id.xy + vec2<i32>(0, -1)) + 0.5) / vec2<f32>(size);
  let uvD: vec2<f32> = (vec2<f32>(id.xy + vec2<i32>(0, 1)) + 0.5) / vec2<f32>(size);

  let dL: f32 = textureLoad(depthTex, vec2<i32>(id.xy + vec2<i32>(-1, 0)), 0).r;
  let dR: f32 = textureLoad(depthTex, vec2<i32>(id.xy + vec2<i32>(1, 0)), 0).r;
  let dU: f32 = textureLoad(depthTex, vec2<i32>(id.xy + vec2<i32>(0, -1)), 0).r;
  let dD: f32 = textureLoad(depthTex, vec2<i32>(id.xy + vec2<i32>(0, 1)), 0).r;

  let pL: vec3<f32> = vec3<f32>((uvL * 2.0 - 1.0) * dL / vec2<f32>(p11, p22), dL);
  let pR: vec3<f32> = vec3<f32>((uvR * 2.0 - 1.0) * dR / vec2<f32>(p11, p22), dR);
  let pU: vec3<f32> = vec3<f32>((uvU * 2.0 - 1.0) * dU / vec2<f32>(p11, p22), dU);
  let pD: vec3<f32> = vec3<f32>((uvD * 2.0 - 1.0) * dD / vec2<f32>(p11, p22), dD);

  let n: vec3<f32> = normalize(cross(pD - pU, pR - pL));
  textureStore(normalTex, vec2<i32>(id.xy), vec4<f32>(n * 0.5 + 0.5, 1.0));
}

// ── SSAO ──
@compute @workgroup_size(16, 16)
fn cs_ssao(@builtin(global_invocation_id) id: vec3<u32>) {
  let size: vec2<u32> = textureDimensions(depthTex);
  if (id.x >= size.x || id.y >= size.y) { return; }

  let uv: vec2<f32> = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(size);
  let depth: f32 = textureLoad(depthTex, vec2<i32>(id.xy), 0).r;
  if (depth >= 1.0) { textureStore(ssaoOut, vec2<i32>(id.xy), vec4<f32>(1.0)); return; }

  let nrm: vec4<f32> = textureLoad(normalTex, vec2<i32>(id.xy), 0);
  let normal: vec3<f32> = normalize(nrm.xyz * 2.0 - 1.0);

  let p11: f32 = proj.p11;
  let p22: f32 = proj.p22;
  let ndc: vec2<f32> = uv * 2.0 - 1.0;
  let vPos: vec3<f32> = vec3<f32>(ndc * depth / vec2<f32>(p11, p22), depth);

  // Tangent frame
  let T: vec3<f32> = normalize(cross(normal, vec3<f32>(0.0, 0.0, 1.0)));
  let B: vec3<f32> = cross(normal, T);
  let rotM: mat3x3<f32> = mat3x3<f32>(T, B, normal);

  let radius: f32 = params.radius;
  let bias: f32 = params.bias;
  let R: f32 = params.radius;
  let texel: vec2<f32> = 1.0 / vec2<f32>(size);

  var occ: f32 = 0.0;
  // 8 hemisphere samples
  for (var s: u32 = 0u; s < 8u; s++) {
    let u1: f32 = f32(s) / 8.0;
    let u2: f32 = f32((s * 251u + 137u) & 0xFFu) / 256.0;
    let theta: f32 = 6.28318 * u1;
    let phi: f32 = acos(u2);
    let sd: vec3<f32> = vec3<f32>(sin(phi) * cos(theta), sin(phi) * sin(theta), cos(phi));
    let spos: vec3<f32> = vPos + rotM * sd * radius;
    let suv: vec2<f32> = vec2<f32>(spos.xy * vec2<f32>(p11, p22) / spos.z);
    let sdepth: f32 = textureLoad(depthTex, vec2<i32>(suv * vec2<f32>(size)), 0).r;
    let range: f32 = smoothstep(0.0, 1.0, R / abs(vPos.z - sdepth));
    occ += select(1.0, 0.0, sdepth >= (spos.z - bias)) * range;
  }
  occ = 1.0 - occ / 8.0;
  occ = pow(max(occ, 0.0), params.power);
  textureStore(ssaoOut, vec2<i32>(id.xy), vec4<f32>(occ));
}

// ── Blur (ping-pong: read from ssaoRaw, write to ssaoOut) ──
@group(0) @binding(5) var ssaoRaw: texture_2d<f32>;

@compute @workgroup_size(16, 16)
fn cs_blur(@builtin(global_invocation_id) id: vec3<u32>) {
  let size: vec2<u32> = textureDimensions(ssaoRaw);
  if (id.x >= size.x || id.y >= size.y) { return; }
  let texel: vec2<f32> = 1.0 / vec2<f32>(size);
  var sum: f32 = 0.0;
  var w: f32 = 0.0;
  for (var dy: i32 = -2; dy <= 2; dy++) {
    for (var dx: i32 = -2; dx <= 2; dx++) {
      let wgt: f32 = 1.0;
      sum += textureLoad(ssaoRaw, vec2<i32>(id.xy) + vec2<i32>(dx, dy), 0).r * wgt;
      w += wgt;
    }
  }
  textureStore(ssaoOut, vec2<i32>(id.xy), vec4<f32>(sum / w));
}
