// ── Specular Prefilter ──
// Convolves the environment cubemap into a prefiltered mip chain
// using GGX importance sampling with Hammersley sequence.
// One mip level per render pass — roughness = mip / (maxLevel - 1).
//
// Input: environment cubemap (texture_cube<f32>), sampler,
//        MVP uniform, roughness uniform
// Output: rgba16float cubemap face at current mip level

@group(0) @binding(0) var<uniform> mvp: mat4x4<f32>;
@group(0) @binding(1) var<uniform> roughness: f32;
@group(0) @binding(2) var envMap: texture_cube<f32>;
@group(0) @binding(3) var envSampler: sampler;

// Include shared helpers at build time via string concat
// hammersley, importanceSampleGGX

const SAMPLE_COUNT: u32 = 4096u;
const CUBEMAP_RES: f32 = 512.0; // source cubemap resolution (per face)

// PDF-based mip level selection (ref: LearnOpenGL, Karis 2013)
// Computes the appropriate LOD for sampling the source environment map
// based on the sample's PDF — clustered (high-PDF) samples read higher LODs
// to prevent aliasing when many samples hit nearby texels.
fn computeMipLevel(pdf: f32, roughness: f32) -> f32 {
  let saTexel: f32 = 4.0 * PI_IBL / (6.0 * CUBEMAP_RES * CUBEMAP_RES);
  let saSample: f32 = 1.0 / (f32(SAMPLE_COUNT) * pdf + 0.0001);
  return select(0.5 * log2(saSample / saTexel), 0.0, roughness == 0.0);
}

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
};

@vertex
fn vs_main(@location(0) position: vec4<f32>) -> VSOut {
  var out: VSOut;
  out.pos = mvp * position;
  out.worldPos = position.xyz;
  return out;
}

@fragment
fn fs_main(@location(0) worldPos: vec3<f32>) -> @location(0) vec4<f32> {
  let N: vec3<f32> = normalize(worldPos);
  // Simplifying assumption: V = R = N
  let R: vec3<f32> = N;
  let V: vec3<f32> = R;

  var prefiltered: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
  var totalWeight: f32 = 0.0;

  for (var i: u32 = 0u; i < SAMPLE_COUNT; i++) {
    let xi: vec2<f32> = hammersley(i, SAMPLE_COUNT);
    let H: vec3<f32> = importanceSampleGGX(xi, N, roughness);
    let L: vec3<f32> = normalize(2.0 * dot(V, H) * H - V);

    let NdotL: f32 = max(dot(N, L), 0.0);
    if (NdotL > 0.0) {
      // PDF-based mip level: clustered (high-PDF) samples read higher LODs
      // to prevent aliasing — the core of proper specular prefiltering
      let NdotH: f32 = max(dot(N, H), 0.0);
      let HdotV: f32 = max(dot(H, V), 0.0);
      let pdf: f32 = distributionGGX(NdotH, roughness) * NdotH / (4.0 * HdotV) + 0.0001;
      let mipLevel: f32 = computeMipLevel(pdf, roughness);

      prefiltered += textureSampleLevel(envMap, envSampler, L, mipLevel).rgb * NdotL;
      totalWeight += NdotL;
    }
  }

  prefiltered = prefiltered / max(totalWeight, 0.001);
  return vec4<f32>(prefiltered, 1.0);
}
