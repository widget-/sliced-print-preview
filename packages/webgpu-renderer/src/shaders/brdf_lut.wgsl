// ── BRDF Integration LUT (split-sum second half) ──
//
// Purpose:
//   Pre-computes the environment BRDF integral for the split-sum
//   approximation. The result is a 512×512 rg16float 2D texture
//   where:
//     R channel = scale  (F₀ coefficient)
//     G channel = bias   (Fresnel-independent term)
//
//   At runtime, the specular IBL is reconstructed as:
//     specularIBL = prefilteredEnv × (F₀ × scale + bias)
//
// Algorithm:
//   For each texel (NdotV, roughness):
//     1. Construct V from NdotV (V lies in the XZ plane for arbitrary tangent frame)
//     2. N = (0, 0, 1) — view-space normal, independent of world orientation
//     3. For SAMPLE_COUNT = 1024 importance-sampled GGX directions:
//        a. Importance-sample H from the GGX distribution at given roughness
//        b. Reflect V about H to get the incident light direction L
//        c. Compute the Smith-GGX geometry visibility:
//             G_Vis = (G × VdotH) / (NdotH × NdotV)
//        d. Accumulate scale and bias weighted by the Fresnel term:
//             Fc = pow(1 - VdotH, 5)
//             A += (1 - Fc) × G_Vis     (scale — F₀ coefficient)
//             B += Fc × G_Vis           (bias  — Fresnel-independent)
//     4. Normalize by sample count
//
//   This integral is environment-independent — it depends only on
//   NdotV and roughness. It can be pre-computed once and reused
//   for any environment map.
//
// Coordinate system:
//   The computation is done in a LOCAL frame where N = (0, 0, 1),
//   independent of the world's Z-up or Y-up convention. Only dot
//   products matter, so the result is orientation-independent.
//   The importanceSampleGGX function (imported from ibl_shared.wgsl)
//   builds a tangent basis from N in this local frame, which is
//   numerically well-conditioned since N = (0, 0, 1) here.
//
// Precision / format:
//   Texture format: rg16float (2-channel, 16-bit float per channel)
//   Size: 512×512
//   uv.x = NdotV (view angle, 0 = grazing, 1 = normal-incidence)
//   uv.y = roughness (perceptual slider value, 0 = smooth, 1 = rough)
//
//   rg16float is filterable in WebGPU (sampleType: 'float'),
//   so textureSample with a linear sampler works correctly.
//
// Shared functions (concatenated from ibl_shared.wgsl):
//   hammersley(i, N), importanceSampleGGX(xi, N, roughness),
//   geometrySmith_IBL(N, V, L, roughness),
//   geometrySchlickGGX_IBL(NdotV, roughness)

@group(0) @binding(0) var<uniform> scale: f32; // unused placeholder (reserved)

const PI_BRDF: f32 = 3.14159265359;
const SAMPLE_COUNT: u32 = 1024u;

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@location(0) position: vec3<f32>, @location(1) uv_in: vec2<f32>) -> VSOut {
  var out: VSOut;
  out.pos = vec4<f32>(position, 1.0);
  out.uv = uv_in;
  return out;
}

// Integrate the BRDF for one texel (NdotV, roughness).
// Returns (scale, bias) for the split-sum approximation.
fn integrateBRDF(NdotV: f32, roughness: f32) -> vec2<f32> {
  // View direction in the local frame: N = (0, 0, 1)
  // V lies in the XZ plane: X = sqrt(1 - NdotV²), Z = NdotV
  let V: vec3<f32> = vec3<f32>(sqrt(1.0 - NdotV * NdotV), 0.0, NdotV);
  let N: vec3<f32> = vec3<f32>(0.0, 0.0, 1.0);

  var A: f32 = 0.0;  // accumulates the F₀ coefficient (scale)
  var B: f32 = 0.0;  // accumulates the Fresnel-independent term (bias)

  for (var i: u32 = 0u; i < SAMPLE_COUNT; i++) {
    let Xi: vec2<f32> = hammersley(i, SAMPLE_COUNT);
    let H: vec3<f32> = importanceSampleGGX(Xi, N, roughness);
    let L: vec3<f32> = normalize(2.0 * dot(V, H) * H - V);

    let NdotL: f32 = max(L.z, 0.0);
    let NdotH: f32 = max(H.z, 0.0);
    let VdotH: f32 = max(dot(V, H), 0.0);

    if (NdotL > 0.0) {
      let G: f32 = geometrySmith_IBL(N, V, L, roughness);
      let G_Vis: f32 = (G * VdotH) / max(NdotH * NdotV, 0.0001);
      let Fc: f32 = pow(1.0 - VdotH, 5.0);      // Schlick Fresnel weight
      A += (1.0 - Fc) * G_Vis;                   // F₀-scaled contribution
      B += Fc * G_Vis;                            // Fresnel-independent contribution
    }
  }

  return vec2<f32>(A / f32(SAMPLE_COUNT), B / f32(SAMPLE_COUNT));
}

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec2<f32> {
  // uv.x = NdotV, uv.y = roughness (linear mapping)
  let result: vec2<f32> = integrateBRDF(uv.x, uv.y);
  return result;
}
