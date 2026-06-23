struct IndirectDraw {
  indexCount: u32,
  instanceCount: u32,
  firstIndex: u32,
  baseVertex: i32,
  firstInstance: u32,
};

struct Camera {
  viewProj: mat4x4<f32>,
  camPos: vec3<f32>,
};

struct SegmentData {
  startPos: vec4<f32>,
  endPos: vec4<f32>,
  chain: vec4<f32>,
  misc: vec4<f32>,
};

struct CullParams {
  vpHeight: f32,
  tanFovHalf: f32,
};

@group(0) @binding(0) var<storage, read> segments: array<SegmentData>;
@group(0) @binding(1) var<uniform> camera: Camera;
@group(0) @binding(2) var<storage, read_write> indirectDraws: array<IndirectDraw, 3>;
@group(0) @binding(3) var<storage, read_write> segmentLod: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> lodCounters: array<atomic<u32>, 3>;
@group(0) @binding(5) var<uniform> cullParams: CullParams;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i: u32 = id.x;
  let count: u32 = arrayLength(&segmentLod);
  if (i >= count) { return; }

  let data: SegmentData = segments[i];
  let start: vec3<f32> = data.startPos.xyz;
  let end: vec3<f32> = data.endPos.xyz;
  let mid: vec4<f32> = vec4<f32>((start + end) * 0.5, 1.0);

  // Compute depth in view space from the view-projection matrix
  // clipZ = viewProj[2].x * mid.x + viewProj[2].y * mid.y + viewProj[2].z * mid.z + viewProj[2].w
  // For a standard projection, this gives clip-space Z. Convert to view Z:
  // viewZ = clipZ / mid_after_proj.w... actually just use the row-approach
  
  // View-space depth from clip space (clip.w = -viewZ)
  let clip: vec4<f32> = camera.viewProj * mid;
  let depth: f32 = abs(clip.w);

  // Screen-space size
  let segLen: f32 = distance(start, end);
  let segWidth: f32 = data.startPos.w;
  let screenSize: f32 = max(segLen, segWidth) * cullParams.vpHeight / (2.0 * depth * cullParams.tanFovHalf);

  var lod: u32;
  if (screenSize < 3.0) {
    lod = 2u;
  } else if (screenSize < 15.0) {
    lod = 1u;
  } else {
    lod = 0u;
  }

  atomicStore(&segmentLod[i], lod);
  atomicAdd(&lodCounters[lod], 1u);
}
