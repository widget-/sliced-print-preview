// ── PBR Fragment Shader (main surface lighting) ──
//
// Purpose:
//   Computes the final lit color for each fragment using a Cook-Torrance
//   microfacet BRDF with GGX distribution, Smith geometry, Schlick Fresnel,
//   and the split-sum IBL approximation for environment lighting.
//   Supports TWO directional lights with independent shadow maps.
//
// Outputs:
//   @location(0): vec4<f32> — lit color (linear RGB in the canvas format)
//   @location(1): vec4<f32> — view-space normal packed in [0, 1] for SSAO
//
// Bind groups:
//   group(0): camera, material, segments, colors, light, caps, LOD — see types.wgsl
//   group(1): shadow map (depth_2d), comparison sampler, light VP matrix
//   group(2): IBL textures (irradianceMap cube, prefilterMap cube, brdfLUT, sampler)
//   group(3): secondary shadow map + light dir2

struct FragOutput {
  @location(0) color: vec4<f32>,
  @location(1) normal: vec4<f32>,
};

// ── Shadow helpers (PCF — rotated Vogel disk + receiver-plane bias) ──

fn interleavedGradientNoise(pos: vec2<f32>) -> f32 {
  return fract(52.9829189 * fract(dot(pos, vec2<f32>(0.06711056, 0.00583715))));
}

fn vogelDiskSample(index: u32, count: u32, phi: f32) -> vec2<f32> {
  let goldenAngle: f32 = 2.399963229728653;
  let r: f32 = sqrt((f32(index) + 0.5) / f32(count));
  let theta: f32 = f32(index) * goldenAngle + phi;
  return vec2<f32>(cos(theta), sin(theta)) * r;
}

fn computeReceiverPlaneDepthBias(p: vec3<f32>) -> vec2<f32> {
  let duvz_dx = dpdx(p);
  let duvz_dy = dpdy(p);
  let inv_det = 1.0 / (duvz_dx.x * duvz_dy.y - duvz_dx.y * duvz_dy.x);
  return vec2<f32>(
    duvz_dy.y * duvz_dx.z - duvz_dx.y * duvz_dy.z,
    duvz_dx.x * duvz_dy.z - duvz_dy.x * duvz_dx.z
  ) * inv_det;
}

fn computeShadow(vp: mat4x4<f32>, tex: texture_depth_2d, smp: sampler_comparison, worldPos: vec3<f32>, clipPos: vec4<f32>) -> f32 {
  let shadowClip: vec4<f32> = vp * vec4<f32>(worldPos, 1.0);
  let shadowNDC: vec3<f32> = shadowClip.xyz / shadowClip.w;
  let shadowUV: vec2<f32> = shadowNDC.xy * vec2<f32>(0.5, -0.5) + 0.5;

  let inFrustum: bool = all(shadowUV == clamp(shadowUV, vec2<f32>(0.0), vec2<f32>(1.0)))
                     && shadowNDC.z >= 0.0 && shadowNDC.z <= 1.0;
  let dz_duv: vec2<f32> = computeReceiverPlaneDepthBias(shadowNDC);

  var vis: f32 = 1.0;
  if (inFrustum) {
    let texelSize: f32 = 1.0 / 1024.0;
    let radius: f32 = texelSize * shadowParams.x;
    let phi: f32 = interleavedGradientNoise(clipPos.xy) * 6.283185307;
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < 12u; i++) {
      let offset: vec2<f32> = vogelDiskSample(i, 12u, phi) * radius;
      let uv: vec2<f32> = clamp(shadowUV + offset, vec2<f32>(0.0), vec2<f32>(1.0));
      let perSampleBias: f32 = dot(dz_duv, offset);
      let refZ: f32 = clamp(shadowNDC.z + perSampleBias, 0.0, 1.0);
      sum += textureSampleCompareLevel(tex, smp, uv, refZ);
    }
    vis = sum / 12.0;
  }
  return vis;
}

// Evaluate a single directional light (specular + diffuse)
fn evalLight(L: vec3<f32>, N: vec3<f32>, V: vec3<f32>, F: vec3<f32>, alpha2: f32, kD: vec3<f32>, baseColor: vec3<f32>, intensity: f32, shadowVis: f32, specularStrength: f32) -> vec3<f32> {
  let NdotV: f32 = max(dot(N, V), 0.0001);
  let NdotL: f32 = max(dot(N, L), 0.0001);
  let H: vec3<f32> = normalize(L + V);
  let NdotH: f32 = max(dot(N, H), 0.0001);
  let NdotH2: f32 = NdotH * NdotH;

  let denom: f32 = NdotH2 * (alpha2 - 1.0) + 1.0;
  let D: f32 = alpha2 / (3.14159265 * denom * denom);

  let a2_NDL: f32 = alpha2 + (1.0 - alpha2) * NdotL * NdotL;
  let G1_l: f32 = 2.0 * NdotL / max(NdotL + sqrt(a2_NDL), 0.0001);
  let a2_NDV: f32 = alpha2 + (1.0 - alpha2) * NdotV * NdotV;
  let G1_v: f32 = 2.0 * NdotV / max(NdotV + sqrt(a2_NDV), 0.0001);

  let spec: vec3<f32> = D * F * G1_l * G1_v / (4.0 * NdotL * NdotV + 0.0001) * specularStrength;
  let diff: vec3<f32> = baseColor * kD / 3.14159265;
  return (diff + spec) * NdotL * intensity * shadowVis;
}

@fragment
fn fs_main(in: VertexOutput) -> FragOutput {
  let N: vec3<f32> = normalize(in.worldNormal);
  let V: vec3<f32> = normalize(camera.camPos - in.worldPos);
  let NdotV: f32 = max(dot(N, V), 0.0001);

  let f0: vec3<f32> = mix(vec3<f32>(0.04), in.color, material.metalness);
  let F: vec3<f32> = f0 + (1.0 - f0) * pow(1.0 - NdotV, 5.0);
  let alpha: f32 = max(material.roughness * material.roughness, 0.001);
  let alpha2: f32 = alpha * alpha;
  let kD: vec3<f32> = (1.0 - F) * (1.0 - material.metalness);

  let shadowVis1: f32 = computeShadow(shadowVP, shadowTex, shadowSampler, in.worldPos, in.clipPos);
  let shadowVis2: f32 = computeShadow(shadowVP2, shadowTex2, shadowSampler2, in.worldPos, in.clipPos);

  let L1: vec3<f32> = normalize(lightDir.xyz);
  let L2: vec3<f32> = normalize(lightDir2.xyz);
  let directSum: vec3<f32> = evalLight(L1, N, V, F, alpha2, kD, in.color, lightDir.w, shadowVis1, material.specularStrength)
                            + evalLight(L2, N, V, F, alpha2, kD, in.color, lightDir2.w, shadowVis2, material.specularStrength);

  // ── IBL (split-sum approximation) ──
  let irradiance: vec3<f32> = textureSample(irradianceMap, iblSampler, N).rgb;
  let diffuseIBL: vec3<f32> = in.color * kD * irradiance * material.envIntensity;

  let R: vec3<f32> = reflect(-V, N);
  let roughnessLOD: f32 = material.roughness * material.roughness * 7.0;
  let lodBase: f32 = floor(roughnessLOD);
  let lodFrac: f32 = roughnessLOD - lodBase;
  let lodHi: f32 = min(lodBase + 1.0, 7.0);
  let prefilteredLo: vec3<f32> = textureSampleLevel(prefilterMap, iblSampler, R, lodBase).rgb;
  let prefilteredHi: vec3<f32> = textureSampleLevel(prefilterMap, iblSampler, R, lodHi).rgb;
  let prefiltered: vec3<f32> = mix(prefilteredLo, prefilteredHi, lodFrac);
  let brdf: vec2<f32> = textureSample(brdfLUT, iblSampler, vec2<f32>(NdotV, material.roughness)).rg;
  let specularIBL: vec3<f32> = prefiltered * (f0 * brdf.x + brdf.y) * material.envIntensity * max(1.0 - material.roughness, 0.0);

  let ambientIBL: vec3<f32> = (diffuseIBL + specularIBL) * material.ambientStrength;

  // Combine: ambient IBL + direct lighting from both lights
  let lit: vec3<f32> = ambientIBL + directSum;

  let viewN: vec3<f32> = (camera.viewMat * vec4<f32>(N, 0.0)).xyz;
  return FragOutput(vec4<f32>(lit, 1.0), vec4<f32>(viewN * 0.5 + 0.5, 1.0));
}
