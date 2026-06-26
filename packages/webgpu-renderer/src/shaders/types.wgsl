const PI: f32 = 3.14159265359;

struct Camera {
  viewProj: mat4x4<f32>,
  viewMat: mat4x4<f32>,
  camPos: vec3<f32>,
};

struct Material {
  roughness: f32,
  metalness: f32,
  envIntensity: f32,
  specularStrength: f32,
  ambientStrength: f32,
  baseColorTint: vec3<f32>,
  _pad: f32,
};

struct SegmentData {
  startPos: vec4<f32>,
  endPos: vec4<f32>,
  chain: vec4<f32>,
  misc: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> material: Material;
@group(0) @binding(2) var<storage, read> segments: array<SegmentData>;
@group(0) @binding(3) var<storage, read> colors: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> lightDir: vec4<f32>;
@group(0) @binding(5) var<storage, read> capInstances: array<vec2<f32>>;
@group(0) @binding(6) var<storage, read> segmentLod: array<u32>;
@group(0) @binding(7) var<uniform> lodLevel: u32;

// Shadow bindings (group 1)
@group(1) @binding(0) var shadowTex: texture_depth_2d;
@group(1) @binding(1) var shadowSampler: sampler_comparison;
@group(1) @binding(2) var<uniform> shadowVP: mat4x4<f32>;

// IBL bindings (group 2)
@group(2) @binding(0) var irradianceMap: texture_cube<f32>;
@group(2) @binding(1) var prefilterMap: texture_cube<f32>;
@group(2) @binding(2) var brdfLUT: texture_2d<f32>;
@group(2) @binding(3) var iblSampler: sampler;

struct VertexInput {
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
};

struct VertexOutput {
  @builtin(position) clipPos: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
  @location(1) worldNormal: vec3<f32>,
  @location(2) color: vec3<f32>,
};
