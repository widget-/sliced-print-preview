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

  // Fresnel (Schlick) — colored F0: gray 0.04 for dielectrics, albedo for metals
  let f0: vec3<f32> = mix(vec3<f32>(0.04), in.color, material.metalness);
  let F: vec3<f32> = f0 + (1.0 - f0) * pow(1.0 - NdotV, 5.0);

  // Roughness
  let alpha: f32 = max(material.roughness * material.roughness, 0.001);
  let alpha2: f32 = alpha * alpha;

  // GGX normal distribution
  let NdotH2: f32 = NdotH * NdotH;
  let denom: f32 = NdotH2 * (alpha2 - 1.0) + 1.0;
  let D: f32 = alpha2 / (3.14159265 * denom * denom);

  // Smith geometry (GGX correlated)
  let a2_NDL: f32 = alpha2 + (1.0 - alpha2) * NdotL * NdotL;
  let G1_l: f32 = 2.0 * NdotL / max(NdotL + sqrt(a2_NDL), 0.0001);
  let a2_NDV: f32 = alpha2 + (1.0 - alpha2) * NdotV * NdotV;
  let G1_v: f32 = 2.0 * NdotV / max(NdotV + sqrt(a2_NDV), 0.0001);

  // Direct specular (Cook-Torrance with colored Fresnel)
  let directSpecular: vec3<f32> = D * F * G1_l * G1_v / (4.0 * NdotL * NdotV + 0.0001) * material.specularStrength;

  // Direct diffuse (Lambertian, energy-conserving via Fresnel)
  let kD: vec3<f32> = (1.0 - F) * (1.0 - material.metalness);
  let directDiffuse: vec3<f32> = in.color * kD / 3.14159265;

  let lightIntensity: f32 = lightDir.w;

  // Shadow mapping (PCF, 5×5 tap) — reference: CSMShadowShaderUtils (MIT, webgpu-sponza-demo)
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

  // IBL — split-sum approximation
  // Diffuse: irradiance map sampled by normal, modulated by albedo
  let irradiance: vec3<f32> = textureSample(irradianceMap, iblSampler, N).rgb;
  let diffuseIBL: vec3<f32> = in.color * kD * irradiance * material.envIntensity;

  // Specular: prefiltered env map at roughness-dependent LOD × DFG LUT (colored Fresnel via F0)
  let R: vec3<f32> = reflect(-V, N);
  let roughnessLOD: f32 = material.roughness * material.roughness * 7.0; // squared = perceptual, matches GGX alpha
  // Manual trilinear blend between adjacent mip levels for portability
  // (textureSampleLevel with fractional LOD doesn't always honor mipmapFilter on all backends)
  let lodBase: f32 = floor(roughnessLOD);
  let lodFrac: f32 = roughnessLOD - lodBase;
  let lodHi: f32 = min(lodBase + 1.0, 7.0);
  let prefilteredLo: vec3<f32> = textureSampleLevel(prefilterMap, iblSampler, R, lodBase).rgb;
  let prefilteredHi: vec3<f32> = textureSampleLevel(prefilterMap, iblSampler, R, lodHi).rgb;
  let prefiltered: vec3<f32> = mix(prefilteredLo, prefilteredHi, lodFrac);
  let brdf: vec2<f32> = textureSample(brdfLUT, iblSampler, vec2<f32>(NdotV, material.roughness)).rg;
  let specularIBL: vec3<f32> = prefiltered * (f0 * brdf.x + brdf.y) * material.envIntensity;

  let ambientIBL: vec3<f32> = diffuseIBL + specularIBL;

  // Albedo applied to diffuse terms only; F0/F handles colored specular for metals
  let lit: vec3<f32> = ambientIBL + (directDiffuse + directSpecular) * NdotL * lightIntensity * shadowVis;

  // Output view-space normal packed in [0,1] for G-buffer SSAO
  let viewN: vec3<f32> = (camera.viewMat * vec4<f32>(N, 0.0)).xyz;
  return FragOutput(vec4<f32>(lit, 1.0), vec4<f32>(viewN * 0.5 + 0.5, 1.0));
}
