struct SSAOParams {
  radius: f32,
  intensity: f32,
  bias: f32,
  power: f32,
  _pad: f32,
};

@group(0) @binding(0) var depthTex: texture_2d<f32>;
@group(0) @binding(1) var normalTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var ssaoTex: texture_storage_2d<r8unorm, write>;
@group(0) @binding(3) var<uniform> ssaoParams: SSAOParams;
@group(0) @binding(4) var<uniform> proj: vec4<f32>; // x = proj._11, y = proj._22, z = near, w = far

// ── Normal reconstruction from depth ──
@compute @workgroup_size(16, 16)
fn cs_normal(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) num: vec3<u32>) {
  let size: vec2<u32> = textureDimensions(depthTex);
  if (id.x >= size.x || id.y >= size.y) { return; }

  let uv: vec2<f32> = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(size);
  let depth: f32 = textureLoad(depthTex, vec2<i32>(id.xy), 0).r;

  // Reconstruct view-space position
  let ndc: vec2<f32> = uv * 2.0 - 1.0;
  let viewPos: vec3<f32> = vec3<f32>(ndc * depth / vec2<f32>(proj.x, proj.y), depth);

  // Compute normal from screen-space derivatives of position
  let texelSize: vec2<f32> = 1.0 / vec2<f32>(size);
  let dLeft: f32 = textureLoad(depthTex, vec2<i32>(id.xy + vec2<i32>(-1, 0)), 0).r;
  let dRight: f32 = textureLoad(depthTex, vec2<i32>(id.xy + vec2<i32>(1, 0)), 0).r;
  let dUp: f32 = textureLoad(depthTex, vec2<i32>(id.xy + vec2<i32>(0, -1)), 0).r;
  let dDown: f32 = textureLoad(depthTex, vec2<i32>(id.xy + vec2<i32>(0, 1)), 0).r;

  let uvL: vec2<f32> = (vec2<f32>(id.xy + vec2<i32>(-1, 0)) + 0.5) / vec2<f32>(size);
  let uvR: vec2<f32> = (vec2<f32>(id.xy + vec2<i32>(1, 0)) + 0.5) / vec2<f32>(size);
  let uvU: vec2<f32> = (vec2<f32>(id.xy + vec2<i32>(0, -1)) + 0.5) / vec2<f32>(size);
  let uvD: vec2<f32> = (vec2<f32>(id.xy + vec2<i32>(0, 1)) + 0.5) / vec2<f32>(size);

  let pL: vec3<f32> = vec3<f32>((uvL * 2.0 - 1.0) * dLeft / vec2<f32>(proj.x, proj.y), dLeft);
  let pR: vec3<f32> = vec3<f32>((uvR * 2.0 - 1.0) * dRight / vec2<f32>(proj.x, proj.y), dRight);
  let pU: vec3<f32> = vec3<f32>((uvU * 2.0 - 1.0) * dUp / vec2<f32>(proj.x, proj.y), dUp);
  let pD: vec3<f32> = vec3<f32>((uvD * 2.0 - 1.0) * dDown / vec2<f32>(proj.x, proj.y), dDown);

  let dx: vec3<f32> = pR - pL;
  let dy: vec3<f32> = pD - pU;
  var n: vec3<f32> = normalize(cross(dy, dx));
  n = select(n, -n, n.z < 0.0); // face toward viewer

  textureStore(normalTex, vec2<i32>(id.xy), vec4<f32>(n * 0.5 + 0.5, 1.0));
}

// ── SSAO ──
@compute @workgroup_size(16, 16)
fn cs_ssao(@builtin(global_invocation_id) id: vec3<u32>) {
  let size: vec2<u32> = textureDimensions(depthTex);
  if (id.x >= size.x || id.y >= size.y) { return; }

  let uv: vec2<f32> = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(size);
  let depth: f32 = textureLoad(depthTex, vec2<i32>(id.xy), 0).r;
  let nrmRaw: vec4<f32> = textureLoad(normalTex, vec2<i32>(id.xy), 0);
  let normal: vec3<f32> = normalize(nrmRaw.xyz * 2.0 - 1.0);

  if (depth >= 1.0) { textureStore(ssaoTex, vec2<i32>(id.xy), vec4<f32>(1.0)); return; }

  // View-space position
  let ndc: vec2<f32> = uv * 2.0 - 1.0;
  let viewPos: vec3<f32> = vec3<f32>(ndc * depth / vec2<f32>(proj.x, proj.y), depth);

  let radius: f32 = ssaoParams.radius;
  let bias: f32 = ssaoParams.bias;
  let intensity: f32 = ssaoParams.intensity;
  let texelSize: vec2<f32> = 1.0 / vec2<f32>(size);

  let kernelSamples: u32 = 8u;
  var occlusion: f32 = 0.0;

  // Hemisphere sampling
  let T: vec3<f32> = normalize(cross(normal, abs(normal.z) < 0.999 ? vec3<f32>(0.0, 0.0, 1.0) : vec3<f32>(1.0, 0.0, 0.0)));
  let B: vec3<f32> = cross(normal, T);
  let rot: mat3x3<f32> = mat3x3<f32>(T, B, normal);

  for (var s: u32 = 0u; s < kernelSamples; s++) {
    // Stratified random direction in hemisphere
    let u1: f32 = f32(s) / f32(kernelSamples);
    let u2: f32 = f32((s * 127u + 1u) & 0xFFu) / 256.0; // pseudo-random
    let theta: f32 = 2.0 * 3.14159 * u1;
    let phi: f32 = acos(u2);
    let sampleDir: vec3<f32> = vec3<f32>(sin(phi) * cos(theta), sin(phi) * sin(theta), cos(phi));

    let samplePos: vec3<f32> = viewPos + rot * sampleDir * radius;
    let sampleUV: vec2<f32> = vec2<f32>(samplePos.xy * vec2<f32>(proj.x, proj.y) / samplePos.z);
    let sampleDepth: f32 = textureLoad(depthTex, vec2<i32>(sampleUV * vec2<f32>(size)), 0).r;
    let rangeCheck: f32 = smoothstep(0.0, 1.0, radius / abs(viewPos.z - sampleDepth));
    occlusion += select(1.0, 0.0, sampleDepth < samplePos.z - bias) * rangeCheck;
  }

  occlusion /= f32(kernelSamples);
  occlusion = 1.0 - occlusion;
  occlusion = pow(max(occlusion, 0.0), ssaoParams.power);
  textureStore(ssaoTex, vec2<i32>(id.xy), vec4<f32>(occlusion));
}

// ── Blur (simple box blur) ──
@compute @workgroup_size(16, 16)
fn cs_blur(@builtin(global_invocation_id) id: vec3<u32>) {
  let size: vec2<u32> = textureDimensions(ssaoTex);
  if (id.x >= size.x || id.y >= size.y) { return; }

  var sum: f32 = 0.0;
  let radius: i32 = 2; // 5×5 box blur
  var count: f32 = 0.0;
  for (var dy: i32 = -radius; dy <= radius; dy++) {
    for (var dx: i32 = -radius; dx <= radius; dx++) {
      let tc: vec2<i32> = vec2<i32>(id.xy) + vec2<i32>(dx, dy);
      sum += textureLoad(ssaoTex, tc, 0).r;
      count += 1.0;
    }
  }
  textureStore(ssaoTex, vec2<i32>(id.xy), vec4<f32>(sum / count));
}
