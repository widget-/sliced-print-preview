// ── BRDF Integration LUT ──
// Pre-computes the environment BRDF integral (split-sum second half)
// as a function of NdotV (x) and roughness (y) using importance-sampled GGX.
// Output: rg16float texture (scale in R, bias in G)
//
// Reference: LearnOpenGL / Epic Games split-sum approximation

@group(0) @binding(0) var<uniform> scale: f32; // unused, placeholder

// Include shared helpers at build time via string concat
// hammersley, importanceSampleGGX, geometrySmith_IBL

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

fn integrateBRDF(NdotV: f32, roughness: f32) -> vec2<f32> {
  let V: vec3<f32> = vec3<f32>(sqrt(1.0 - NdotV * NdotV), 0.0, NdotV);
  let N: vec3<f32> = vec3<f32>(0.0, 0.0, 1.0);

  var A: f32 = 0.0;
  var B: f32 = 0.0;

  for (var i: u32 = 0u; i < SAMPLE_COUNT; i++) {
    let Xi: vec2<f32> = hammersley(i, SAMPLE_COUNT);
    let H: vec3<f32> = importanceSampleGGX(Xi, roughness);
    let L: vec3<f32> = normalize(2.0 * dot(V, H) * H - V);

    let NdotL: f32 = max(L.z, 0.0);
    let NdotH: f32 = max(H.z, 0.0);
    let VdotH: f32 = max(dot(V, H), 0.0);

    if (NdotL > 0.0) {
      let G: f32 = geometrySmith_IBL(N, V, L, roughness);
      let G_Vis: f32 = (G * VdotH) / max(NdotH * NdotV, 0.0001);
      let Fc: f32 = pow(1.0 - VdotH, 5.0);
      A += (1.0 - Fc) * G_Vis;
      B += Fc * G_Vis;
    }
  }

  return vec2<f32>(A / f32(SAMPLE_COUNT), B / f32(SAMPLE_COUNT));
}

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec2<f32> {
  let result: vec2<f32> = integrateBRDF(uv.x, uv.y);
  return result;
}
