// ── Shared IBL helper functions ──
// Hammersley sequence, ImportanceSampleGGX, GGX distribution, Smith geometry.
// Included (via concat) into prefilter and BRDF LUT shaders.

const PI_IBL: f32 = 3.14159265359;

// Van der Corput radical inverse in base 2
fn radicalInverseVdC(bits_in: u32) -> f32 {
  var bits = bits_in;
  bits = (bits << 16u) | (bits >> 16u);
  bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
  bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
  bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
  bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
  return f32(bits) * 2.3283064365386963e-10; // 0x100000000^{-1}
}

// Hammersley 2D low-discrepancy sequence
fn hammersley(i: u32, N: u32) -> vec2<f32> {
  return vec2<f32>(f32(i) / f32(N), radicalInverseVdC(i));
}

// GGX NDF (Trowbridge-Reitz)
fn distributionGGX(NdotH: f32, roughness: f32) -> f32 {
  let a: f32 = roughness * roughness;
  let a2: f32 = a * a;
  let denom: f32 = NdotH * NdotH * (a2 - 1.0) + 1.0;
  return a2 / (PI_IBL * denom * denom);
}

// GGX importance sampling — returns half-vector H in world space, aligned to normal N
fn importanceSampleGGX(xi: vec2<f32>, N: vec3<f32>, roughness: f32) -> vec3<f32> {
  let a: f32 = roughness * roughness;
  let phi: f32 = 2.0 * PI_IBL * xi.x;
  let cosTheta: f32 = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
  let sinTheta: f32 = sqrt(1.0 - cosTheta * cosTheta);

  // Half-vector H in tangent space (z = up = N)
  let H: vec3<f32> = vec3<f32>(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);

  // Tangent-to-world transform (build orthonormal basis from N)
  let up: vec3<f32> = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), abs(N.z) < 0.999);
  let tangent: vec3<f32> = normalize(cross(up, N));
  let bitangent: vec3<f32> = cross(N, tangent);

  return normalize(tangent * H.x + bitangent * H.y + N * H.z);
}

// Smith geometry function (IBL variant: k = a²/2)
fn geometrySchlickGGX_IBL(NdotV: f32, roughness: f32) -> f32 {
  let a: f32 = roughness * roughness;
  let k: f32 = (a * a) / 2.0;
  return NdotV / (NdotV * (1.0 - k) + k);
}

fn geometrySmith_IBL(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, roughness: f32) -> f32 {
  let NdotV: f32 = max(dot(N, V), 0.0);
  let NdotL: f32 = max(dot(N, L), 0.0);
  return geometrySchlickGGX_IBL(NdotV, roughness) * geometrySchlickGGX_IBL(NdotL, roughness);
}
