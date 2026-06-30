// ── Shared types and bindings (concatenated first into every shader) ──
//
// Coordinate system: WORLD = Z-UP, right-handed.
//   - +Z is "up" / sky direction
//   - -Z is "down" / ground direction
//   - +Y is "north" (horizontal forward in the default view)
//   - +X is "east" (horizontal right in the default view)
//   - Cubemap faces: +X=right, -X=left, +Y=north, -Y=south, +Z=up/sky, -Z=down/ground
//
// Color: all lighting is computed in LINEAR space.
//   - Base color (in.color) arrives in linear (gamma decoded by loader)
//   - HDR environment arrives in linear half-float (rgba16float)
//   - Output is written to an sRGB (or rgba8unorm) framebuffer;
//     the display pipeline applies the sRGB transfer function.
//
// Precision: all HDR textures (environment cubemap, prefilter, irradiance)
// use rgba16float (FP16, ~11-bit mantissa). The BRDF LUT uses rg16float.
// 16-bit float provides sufficient dynamic range and precision for IBL.

const PI: f32 = 3.14159265359;

// ── group(0): Per-frame uniform / storage bindings ──

struct Camera {
  viewProj: mat4x4<f32>,  // view-projection (world → clip), column-major
  viewMat: mat4x4<f32>,   // view matrix only (world → view), column-major
  camPos: vec3<f32>,      // camera position in world space
  // (implicit 4-byte padding to 16-byte struct alignment)
};

struct Material {
  roughness: f32,          // [0,1] — perceptual roughness slider value
  metalness: f32,          // [0,1] — 0=dielectric, 1=metal (F0 = baseColor)
  envIntensity: f32,       // [0,∞] — environment map multiplier
  specularStrength: f32,   // [0,∞] — direct specular (Cook-Torrance) multiplier
  ambientStrength: f32,    // [0,∞] — IBL multiplier (scales diffuse+specular environment)
  arcCurvature: f32,       // [0.1, 1] — arc/conic curvature multiplier (1 = full bend)
  baseColorTint: vec3<f32>,// linear RGB base color (parsed from hex #RRGGBB)
  useRoleColors: f32,      // 1.0 = use per-segment role colors, 0.0 = uniform baseColorTint
};

struct SegmentData {
  startPos: vec4<f32>,     // segment start (xyz=world position, w unused)
  endPos: vec4<f32>,       // segment end   (xyz=world position, w unused)
  chain: vec4<f32>,        // (parentIdx, type, _, _) — LOD chain link
  misc: vec4<f32>,         // (radius, _, _, _)
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> material: Material;
@group(0) @binding(2) var<storage, read> segments: array<SegmentData>;
@group(0) @binding(3) var<storage, read> colors: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> lightDir: vec4<f32>;  // xyz=normalized, w=intensity
@group(0) @binding(5) var<storage, read> capInstances: array<vec2<f32>>;
@group(0) @binding(6) var<storage, read> segmentLod: array<u32>;
@group(0) @binding(7) var<uniform> lodLevel: u32;

// ── group(1): Shadow mapping ──
// Shadow map: 1024×1024 depth texture from a directional light's POV.
// Sampled with PCF (5×5) using a comparison sampler.

@group(1) @binding(0) var shadowTex: texture_depth_2d;
@group(1) @binding(1) var shadowSampler: sampler_comparison;
@group(1) @binding(2) var<uniform> shadowVP: mat4x4<f32>;  // light VP (world → shadow clip)
@group(1) @binding(3) var<uniform> shadowParams: vec4<f32>; // x=softness (kernel radius multiplier)

// ── group(2): IBL (image-based lighting) ──
// Three cubemap textures + one 2D BRDF LUT, all pre-computed at init time.
// Sampler is shared (mag/min=linear, no mipmap filtering needed for explicit-LOD samples).

@group(2) @binding(0) var irradianceMap: texture_cube<f32>;  // 32×32 per face, rgba16float, diffuse hemisphere integral
@group(2) @binding(1) var prefilterMap: texture_cube<f32>;   // 256×256 per face ×8 mips, rgba16float, GGX-importance-sampled specular
@group(2) @binding(2) var brdfLUT: texture_2d<f32>;          // 512×512, rg16float, split-sum scale(R)/bias(G)
@group(2) @binding(3) var iblSampler: sampler;                // magFilter=linear, minFilter=linear, no compare

// ── group(3): Secondary light (fill) shadow + direction ──
// Front-right-up fill light, dimmer than the primary key light.
// Uses the same comparison sampler (shared from group(1)).

@group(3) @binding(0) var shadowTex2: texture_depth_2d;
@group(3) @binding(1) var shadowSampler2: sampler_comparison;
@group(3) @binding(2) var<uniform> shadowVP2: mat4x4<f32>;
@group(3) @binding(3) var<uniform> lightDir2: vec4<f32>;

// ── Vertex input/output ──
// Vertex buffers are interleaved (pos+normal), 24 bytes per vertex.
// worldPos and worldNormal are computed in the vertex shader (segment.wgsl / cap.wgsl).

struct VertexInput {
  @location(0) position: vec3<f32>,   // local-space position (Z-up geometry)
  @location(1) normal: vec3<f32>,     // local-space normal
};

struct VertexOutput {
  @builtin(position) clipPos: vec4<f32>,
  @location(0) worldPos: vec3<f32>,   // world-space position (Z-up)
  @location(1) worldNormal: vec3<f32>,// world-space normal, normalized
  @location(2) color: vec3<f32>,      // per-segment base color (linear RGB)
};
