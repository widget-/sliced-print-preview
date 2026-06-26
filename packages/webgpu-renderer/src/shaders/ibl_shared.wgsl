// ── Shared IBL helper functions ──
// Concatenated (by ibl.ts) before prefilter_envmap.wgsl and brdf_lut.wgsl.
// Also imported separately by other shaders via the string-concat build step.
//
// Functions:
//   hammersley(i, N)                 — Van der Corput + linear, low-discrepancy 2D sequence
//   importanceSampleGGX(xi, N, r)    — GGX importance sampling, returns H in world space
//   distributionGGX(NdotH, r)        — Trowbridge-Reitz / GGX normal distribution
//   geometrySchlickGGX_IBL(NdotV, r) — Smith-GGX geometry term (IBL variant, k = α² / 2)
//   geometrySmith_IBL(N, V, L, r)    — Combined Smith geometry for IBL (G1_view × G1_light)
//
// Roughness convention:
//   All functions take `roughness: f32` as the [0,1] perceptual slider value.
//   Internally they square it to α = roughness² (the GGX roughness parameter).
//   This matches the direct-lighting path in pbr.wgsl (alpha = roughness²).
//   The prefilter mip chain is spaced linearly in perceptual roughness,
//   so the runtime LOD lookup also squares: LOD = roughness² × (levels-1).
//
// Coordinate convention (Z-up world):
//   - importanceSampleGGX: builds a tangent frame from N with Z-up fallback.
//     The tangent-space Z (cosTheta) corresponds to the world-space normal N.
//   - The geometry functions are invariant to the coordinate system;
//     they operate on dot products only (NdotV, NdotL).
//
// Used by:
//   prefilter_envmap.wgsl   — distributionGGX, importanceSampleGGX, hammersley
//   brdf_lut.wgsl           — importanceSampleGGX, geometrySmith_IBL, hammersley
//   pbr.wgsl                — (has its own inline distributionGGX and geometry)

const PI_IBL: f32 = 3.14159265359;

// ── Hammersley 2D low-discrepancy sequence ──
// Input:  i = sample index within [0, N)
//         N = total sample count
// Output: vec2(xi_x, xi_y) in [0,1)²
// The x component is the linear fraction i/N.
// The y component is the radical inverse of i in base 2 (bit-reversal).
fn radicalInverseVdC(bits_in: u32) -> f32 {
  var bits = bits_in;
  bits = (bits << 16u) | (bits >> 16u);
  bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
  bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
  bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
  bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
  return f32(bits) * 2.3283064365386963e-10; // 1/2³²
}
fn hammersley(i: u32, N: u32) -> vec2<f32> {
  return vec2<f32>(f32(i) / f32(N), radicalInverseVdC(i));
}

// ── GGX NDF (Trowbridge-Reitz) ──
// NdotH: cos(angle between macro-surface normal N and micro-surface half-vector H)
// roughness: [0,1] perceptual value (internally squared to α)
//
// Reference: Trowbridge & Reitz 1975, Walter et al. 2007 (GGX / Trowbridge-Reitz)
fn distributionGGX(NdotH: f32, roughness: f32) -> f32 {
  let a: f32 = roughness * roughness;  // α = roughness²
  let a2: f32 = a * a;
  let denom: f32 = NdotH * NdotH * (a2 - 1.0) + 1.0;
  return a2 / (PI_IBL * denom * denom);
}

// ── GGX importance sampling — returns half-vector H in world space ──
// xi:      2D random sample from Hammersley sequence (or other low-discrepancy)
// N:       macro-surface normal in world space (Z-up, normalized)
// roughness: [0,1] perceptual value (internally squared to α)
//
// Returns H (half-vector) in world space, normalized.
// The view reflection is then L = 2 * dot(V, H) * H - V.
//
// Reference: Karis 2013, "Real Shading in Unreal Engine 4"
//   cosTheta = sqrt((1 - ξy) / (1 + (α² - 1) * ξy))
//   This concentrates samples toward N for low α (mirror-like) and spreads
//   them for high α (rough diffuse-like).
fn importanceSampleGGX(xi: vec2<f32>, N: vec3<f32>, roughness: f32) -> vec3<f32> {
  let a: f32 = roughness * roughness;  // α = roughness²
  let phi: f32 = 2.0 * PI_IBL * xi.x;
  let cosTheta: f32 = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
  let sinTheta: f32 = sqrt(1.0 - cosTheta * cosTheta);

  // Half-vector H in tangent space (z = up = N)
  let H: vec3<f32> = vec3<f32>(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);

  // Tangent-to-world transform (build orthonormal basis from N)
  // Uses Z-up fallback (world Z = up). When N is nearly parallel to Z,
  // falls back to using +X as the "up" for the tangent frame construction.
  // Default: world Z = up. When N is nearly parallel to Z (|N.z| ≈ 1),
  // cross(up, N) would be degenerate, so fall back to +X as up.
  let up: vec3<f32> = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(N.z) < 0.999);
  let tangent: vec3<f32> = normalize(cross(up, N));
  let bitangent: vec3<f32> = cross(N, tangent);

  return normalize(tangent * H.x + bitangent * H.y + N * H.z);
}

// ── Smith geometry function (IBL variant) ──
// NdotV: cos(angle between normal and view direction)
// roughness: [0,1] perceptual value (internally squared to α)
//
// IBL variant: k = α² / 2  (where α = roughness²).
// Compare with the direct-lighting variant in pbr.wgsl (k = (α+1)² / 8).
// The IBL variant is derived from integrating over the hemisphere rather
// than evaluating at a single light direction.
//
// Reference: Schlick 1994, Smith 1967, Karis 2013
//   G = NdotV / (NdotV * (1-k) + k)
fn geometrySchlickGGX_IBL(NdotV: f32, roughness: f32) -> f32 {
  let a: f32 = roughness * roughness;  // α = roughness²
  let k: f32 = (a * a) / 2.0;         // k = α² / 2 = roughness⁴ / 2
  return NdotV / (NdotV * (1.0 - k) + k);
}

// ── Smith geometry (combined, IBL variant) ──
// Multiplies the view and light geometry terms.
fn geometrySmith_IBL(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, roughness: f32) -> f32 {
  let NdotV: f32 = max(dot(N, V), 0.0);
  let NdotL: f32 = max(dot(N, L), 0.0);
  return geometrySchlickGGX_IBL(NdotV, roughness) * geometrySchlickGGX_IBL(NdotL, roughness);
}
