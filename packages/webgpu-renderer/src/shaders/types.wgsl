const PI: f32 = 3.14159265359;

struct Camera {
  viewProj: mat4x4<f32>,
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
