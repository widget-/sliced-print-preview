// ── PBR Fragment Shader (main surface lighting) ──
//
// Purpose:
//   Computes the final lit color for each fragment using a Cook-Torrance
//   microfacet BRDF with GGX distribution, Smith geometry, Schlick Fresnel,
//   and the split-sum IBL approximation for environment lighting.
//
// Outputs:
//   @location(0): vec4<f32> — lit color (linear RGB in the canvas format)
//   @location(1): vec4<f32> — view-space normal packed in [0, 1] for SSAO
//
// Coordinate system: Z-up (+Z = sky, -Z = ground)
//   - N, V, L, H are all in world space
//   - Reflect direction R = reflect(-V, N) points into the environment
//   - The prefilter cubemap and irradiance cubemap follow the same Z-up
//     convention (their +Z face = sky direction)
//
// Bind groups:
//   group(0): camera, material, segments, colors, light, caps, LOD — see types.wgsl
//   group(1): shadow map (depth_2d), comparison sampler, light VP matrix
//   group(2): IBL textures (irradianceMap cube, prefilterMap cube ×8 mips,
//             brdfLUT 2d 512×512, shared linear sampler)
//
// Lighting paths:
//   1. Direct specular:  D × F × G / (4 × NdotL × NdotV)   (Cook-Torrance GGX)
//   2. Direct diffuse:   (1 - F) × (1 - metalness) / π      (Lambertian, energy-conserving)
//   3. IBL specular:     prefiltered × (F₀ × scale + bias)  (split-sum)
//   4. IBL diffuse:      kD × irradiance                    (irradiance map × albedo × kD)
//
// Key formulas:
//   Fresnel (Schlick):       F = F₀ + (1 - F₀) × (1 - NdotV)⁵
//     F₀ = mix(0.04, baseColor, metalness) — gray for dielectrics, colored for metals
//   GGX NDF:                 D = α² / (π × (NdotH² × (α² - 1) + 1)²)
//     α = max(roughness², 0.001)
//   Smith GGX (direct):      k = (α + 1)² / 8
//   Energy conservation:     kD = (1 - F) × (1 - metalness)
//     The Fresnel term F accounts for energy that goes into specular reflection;
//     the remaining (1 - F) is available for diffuse, further reduced by (1 - metalness)
//     since metals have no body diffuse.
//
// Albedo handling:
//   Diffuse terms carry the albedo (baseColor): directDiffuse = albedo × kD / π
//   Specular terms for metals use the colored F₀ = albedo
//   The final lit color is NOT multiplied by albedo again (unlike a scalar-F₀ approach)
//
// Roughness LOD mapping (IBL specular):
//   roughnessLOD = roughness² × 7.0
//   This maps the perceptual slider to a GGX-linear index via squared alpha,
//   then scales by (prefilter levels − 1) = 7 for 8 mip levels.
//   A manual trilinear blend between adjacent mip levels is done for portability
//   (not all backends honor mipmapFilter: 'linear' with textureSampleLevel).

struct FragOutput {
  @location(0) color: vec4<f32>,
  @location(1) normal: vec4<f32>,
};

@fragment
fn fs_main(in: VertexOutput) -> FragOutput {
  let N: vec3<f32> = normalize(in.worldNormal);
  let L: vec3<f32> = normalize(lightDir.xyz);
  let V: vec3<f32> = normalize(camera.camPos - in.worldPos);
  let H: vec3<f32> = normalize(L + V);

  let NdotL: f32 = max(dot(N, L), 0.0001);
  let NdotV: f32 = max(dot(N, V), 0.0001);
  let NdotH: f32 = max(dot(N, H), 0.0001);

  // ── Fresnel (Schlick) — colored F0 ──
  // Dielectrics: F₀ = 0.04 (gray, ~2% normal-incidence reflectance)
  // Metals:      F₀ = baseColor (colored specular reflectance)
  let f0: vec3<f32> = mix(vec3<f32>(0.04), in.color, material.metalness);
  let F: vec3<f32> = f0 + (1.0 - f0) * pow(1.0 - NdotV, 5.0);

  // ── Roughness ──
  // α = roughness², clamped to avoid numerical issues at α=0
  let alpha: f32 = max(material.roughness * material.roughness, 0.001);
  let alpha2: f32 = alpha * alpha;

  // ── GGX normal distribution (Trowbridge-Reitz) ──
  let NdotH2: f32 = NdotH * NdotH;
  let denom: f32 = NdotH2 * (alpha2 - 1.0) + 1.0;
  let D: f32 = alpha2 / (3.14159265 * denom * denom);

  // ── Smith geometry (GGX correlated, direct-lighting variant) ──
  // k = (α + 1)² / 8 for direct lighting (cf. IBL variant in ibl_shared.wgsl: k = α² / 2)
  let a2_NDL: f32 = alpha2 + (1.0 - alpha2) * NdotL * NdotL;
  let G1_l: f32 = 2.0 * NdotL / max(NdotL + sqrt(a2_NDL), 0.0001);
  let a2_NDV: f32 = alpha2 + (1.0 - alpha2) * NdotV * NdotV;
  let G1_v: f32 = 2.0 * NdotV / max(NdotV + sqrt(a2_NDV), 0.0001);

  // ── Direct specular (Cook-Torrance with colored Fresnel) ──
  let directSpecular: vec3<f32> = D * F * G1_l * G1_v / (4.0 * NdotL * NdotV + 0.0001) * material.specularStrength;

  // ── Direct diffuse (Lambertian, energy-conserving via Fresnel) ──
  // kD = diffuse fraction: Fresnel takes energy for specular, metalness kills it
  let kD: vec3<f32> = (1.0 - F) * (1.0 - material.metalness);
  let directDiffuse: vec3<f32> = in.color * kD / 3.14159265;

  let lightIntensity: f32 = lightDir.w;

  // ── Shadow mapping (PCF, 5×5 tap) ──
  // Reference: CSMShadowShaderUtils (MIT, webgpu-sponza-demo)
  let shadowClip: vec4<f32> = shadowVP * vec4<f32>(in.worldPos, 1.0);
  let shadowNDC: vec3<f32> = shadowClip.xyz / shadowClip.w;
  // NDC [-1,1] → UV [0,1] with Y flip (WebGPU NDC Y up, texture Y down)
  let shadowUV: vec2<f32> = shadowNDC.xy * vec2<f32>(0.5, -0.5) + 0.5;
  var shadowVis: f32 = 0.0;
  let texel: f32 = 1.0 / 1024.0;
  let bias: f32 = 0.001;
  for (var dy: i32 = -2; dy <= 2; dy++) {
    for (var dx: i32 = -2; dx <= 2; dx++) {
      let uv: vec2<f32> = shadowUV + vec2<f32>(f32(dx), f32(dy)) * texel;
      shadowVis += textureSampleCompareLevel(shadowTex, shadowSampler, uv, shadowNDC.z - bias);
    }
  }
  shadowVis /= 25.0;

  // ── IBL (split-sum approximation) ──
  // Diffuse: irradiance map sampled by normal, modulated by albedo and kD
  let irradiance: vec3<f32> = textureSample(irradianceMap, iblSampler, N).rgb;
  let diffuseIBL: vec3<f32> = in.color * kD * irradiance * material.envIntensity;

  // Specular: prefiltered env map at roughness-dependent LOD × DFG LUT
  // Reflection vector R points into the environment (what the surface reflects)
  let R: vec3<f32> = reflect(-V, N);
  // LOD = roughness² × (levels-1) — squared matches GGX alpha, levels-1 = 7
  let roughnessLOD: f32 = material.roughness * material.roughness * 7.0;
  // Manual trilinear blend between adjacent mip levels for portability
  // (textureSampleLevel with fractional LOD doesn't always honor
  //  mipmapFilter on all backends, so we blend ourselves)
  let lodBase: f32 = floor(roughnessLOD);
  let lodFrac: f32 = roughnessLOD - lodBase;
  let lodHi: f32 = min(lodBase + 1.0, 7.0);
  let prefilteredLo: vec3<f32> = textureSampleLevel(prefilterMap, iblSampler, R, lodBase).rgb;
  let prefilteredHi: vec3<f32> = textureSampleLevel(prefilterMap, iblSampler, R, lodHi).rgb;
  let prefiltered: vec3<f32> = mix(prefilteredLo, prefilteredHi, lodFrac);
  // BRDF LUT: uv.x = NdotV, uv.y = roughness; R = scale, G = bias
  let brdf: vec2<f32> = textureSample(brdfLUT, iblSampler, vec2<f32>(NdotV, material.roughness)).rg;
  let specularIBL: vec3<f32> = prefiltered * (f0 * brdf.x + brdf.y) * material.envIntensity;

  let ambientIBL: vec3<f32> = diffuseIBL + specularIBL;

  // Combine: ambient IBL (always present) + direct light (modulated by shadow)
  let lit: vec3<f32> = ambientIBL + (directDiffuse + directSpecular) * NdotL * lightIntensity * shadowVis;

  // Output view-space normal packed in [0,1] for G-buffer SSAO
  let viewN: vec3<f32> = (camera.viewMat * vec4<f32>(N, 0.0)).xyz;
  return FragOutput(vec4<f32>(lit, 1.0), vec4<f32>(viewN * 0.5 + 0.5, 1.0));
}
