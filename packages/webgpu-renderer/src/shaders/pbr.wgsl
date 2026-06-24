@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let N: vec3<f32> = normalize(in.worldNormal);
  let L: vec3<f32> = normalize(lightDir.xyz);
  let V: vec3<f32> = normalize(camera.camPos - in.worldPos);
  let H: vec3<f32> = normalize(L + V);

  let NdotL: f32 = max(dot(N, L), 0.0001);
  let NdotV: f32 = max(dot(N, V), 0.0001);
  let NdotH: f32 = max(dot(N, H), 0.0001);

  // Fresnel (Schlick)
  let f0: f32 = mix(0.04, 1.0, material.metalness);
  let F: f32 = f0 + (1.0 - f0) * pow(1.0 - NdotV, 5.0);

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

  // Cook-Torrance specular
  let specular: f32 = D * F * G1_l * G1_v / (4.0 * NdotL * NdotV + 0.0001) * material.specularStrength;

  // Diffuse (Lambertian, energy-conserving via Fresnel)
  let diffuse: f32 = (1.0 - F) * (1.0 - material.metalness) / 3.14159265;

  let lightIntensity: f32 = lightDir.w;

  // Shadow mapping (PCF, 5×5 tap) — reference: CSMShadowShaderUtils (MIT, webgpu-sponza-demo)
  let shadowClip: vec4<f32> = shadowVP * vec4<f32>(in.worldPos, 1.0);
  let shadowNDC: vec3<f32> = shadowClip.xyz / shadowClip.w;
  // NDC [-1,1] → UV [0,1] with Y flip (WebGPU NDC Y up, texture Y down)
  let shadowUV: vec2<f32> = shadowNDC.xy * vec2<f32>(0.5, -0.5) + 0.5;
  var shadowVis: f32 = 0.0;
  let texel: f32 = 1.0 / 2048.0;
  let bias: f32 = 0.001;
  for (var dy: i32 = -2; dy <= 2; dy++) {
    for (var dx: i32 = -2; dx <= 2; dx++) {
      let uv: vec2<f32> = shadowUV + vec2<f32>(f32(dx), f32(dy)) * texel;
      shadowVis += textureSampleCompareLevel(shadowTex, shadowSampler, uv, shadowNDC.z - bias);
    }
  }
  shadowVis /= 25.0;

  // Apply shadow to diffuse+specular (ambient unaffected)
  let attenuated: f32 = material.ambientStrength + (diffuse + specular) * NdotL * lightIntensity * shadowVis;
  let lit: vec3<f32> = in.color * attenuated;
  return vec4<f32>(lit, 1.0);
}
