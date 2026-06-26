// ── Specular Prefilter (GGX importance-sampled convolution) ──
//
// Purpose:
//   Convolves the environment cubemap into a mipmapped prefilter cubemap
//   for the split-sum specular IBL approximation. Each mip level corresponds
//   to a different roughness value:
//     mip 0: roughness = 0.0 (perfect mirror reflection)
//     mip i: roughness = i / (levels-1)  (linearly spaced in perceptual roughness)
//     mip 7: roughness = 1.0 (fully diffuse-like spread)
//
//   Levels: 8 (0–7), cube size 256×256 per face, rgba16float.
//
// Algorithm (per mip level, per face):
//   For each pixel (direction N = face normal):
//     1. Generate SAMPLE_COUNT = 4096 Hammersley low-discrepancy samples
//     2. Importance-sample the GGX distribution to get half-vectors H
//     3. Compute the reflection direction L = 2(V·H)H - V (with V = N)
//     4. Sample the source env cubemap at L, selecting a PDF-based mip level
//        to prevent aliasing from clustered samples
//     5. Weight by NdotL, accumulate
//     6. Normalize by total weight
//
// PDF-based mip selection (Karis 2013):
//   computeMipLevel() calculates the solid-angle footprint of each sample
//   and selects a corresponding LOD of the source cubemap. High-PDF samples
//   (tightly clustered near N) read from lower source LODs (sharper), while
//   low-PDF samples (spread across the hemisphere) read from higher source
//   LODs (blurrier) to avoid aliasing. This is essential for correct
//   prefiltering at high roughness values.
//
//   saTexel  = 4π / (6 × cubeRes²)       — solid angle of one source texel
//   saSample = 1 / (N × pdf + ε)         — solid angle this sample represents
//   mipLevel = 0.5 × log2(saSample / saTexel), clamped to [0, maxMip]
//
// Assumptions:
//   V = R = N (the view direction equals the reflection direction equals
//   the face normal). This is the standard simplification for the
//   split-sum prefilter (Karis 2013): the environment cubemap is always
//   sampled with the reflected view direction at runtime, so during
//   prefiltering we assume the most common case V = N.
//
// Input:
//   envMap: texture_cube<f32> — source environment cubemap (512×512, 10 mips, rgba16float)
//   roughness: f32            — perceptual roughness for this mip level [0, 1]
//   mvp: mat4x4<f32>          — face-specific view-projection matrix
//
// Output:
//   rgba16float cubemap face at the current mip level of prefilterMap.
//   Written to binding 1 of the IBL group in pipeline.ts.
//
// Shared functions (concatenated from ibl_shared.wgsl):
//   hammersley(i, N), importanceSampleGGX(xi, N, roughness), distributionGGX(NdotH, roughness)

@group(0) @binding(0) var<uniform> mvp: mat4x4<f32>;
@group(0) @binding(1) var<uniform> roughness: f32;        // perceptual roughness for this mip
@group(0) @binding(2) var envMap: texture_cube<f32>;      // source cubemap (from equirect→cube)
@group(0) @binding(3) var envSampler: sampler;             // linear filtering for source samples

const SAMPLE_COUNT: u32 = 4096u;
const CUBEMAP_RES: f32 = 512.0; // source cubemap resolution (per face, mip 0)

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
  // N = face normal = the direction this pixel represents
  let N: vec3<f32> = normalize(worldPos);

  // Simplifying assumption: V = R = N
  // (the prefilter assumes the view direction equals the reflection
  //  direction equals the face normal — see Karis 2013)
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
