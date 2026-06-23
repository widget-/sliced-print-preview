import { generateBodyGeometry, generateCapGeometry } from './geometry';
import type { GpuSegmentBuffers } from './buffer';
import { OrbitCamera } from './camera';

// Shader source (embedded for simplicity)
const SHADER_SRC = `
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

@vertex
fn vs_main(in: VertexInput, @builtin(instance_index) ii: u32) -> VertexOutput {
  let data = segments[ii];
  let segColor = colors[ii].rgb;

  let t: f32 = in.position.z + 0.5;
  let packed: u32 = u32(data.misc.x);
  let isArc: bool = (packed & 1u) != 0u;
  let width: f32 = data.startPos.w;

  var segPos: vec3<f32>;
  var endTangent: vec3<f32>;

  if (isArc) {
    // Rational quadratic Bézier: P0 = start, P1 = end (control), P2 = next.start
    let p0: vec3<f32> = data.startPos.xyz;
    let p1: vec3<f32> = data.endPos.xyz;
    let p2: vec3<f32> = segments[ii + 1u].startPos.xyz;
    let w: f32 = data.endPos.w;
    let mt: f32 = 1.0 - t;
    let mt2: f32 = mt * mt;
    let t2: f32 = t * t;
    let denom: f32 = mt2 + 2.0 * t * mt * w + t2;
    segPos = (mt2 * p0 + 2.0 * t * mt * w * p1 + t2 * p2) / denom;

    // Finite-difference tangent for endTangent
    let eps: f32 = 0.01;
    let te: f32 = min(t + eps, 1.0); let me: f32 = 1.0 - te;
    let me2: f32 = me * me; let te2: f32 = te * te;
    let de: f32 = me2 + 2.0 * te * me * w + te2;
    let pe: vec3<f32> = (me2 * p0 + 2.0 * te * me * w * p1 + te2 * p2) / de;
    let ts: f32 = max(t - eps, 0.0); let ms: f32 = 1.0 - ts;
    let ms2: f32 = ms * ms; let ts2: f32 = ts * ts;
    let ds: f32 = ms2 + 2.0 * ts * ms * w + ts2;
    let ps: vec3<f32> = (ms2 * p0 + 2.0 * ts * ms * w * p1 + ts2 * p2) / ds;
    let dDir: vec3<f32> = pe - ps;
    let dLen: f32 = length(dDir);
    endTangent = select(normalize(dDir), vec3<f32>(0.0, 0.0, 1.0), dLen < 0.0001);
  } else {
    segPos = mix(data.startPos.xyz, data.endPos.xyz, t);
    let dir: vec3<f32> = data.endPos.xyz - data.startPos.xyz;
    let segLen: f32 = length(dir);
    endTangent = select(dir / segLen, vec3<f32>(0.0, 0.0, 1.0), segLen < 0.001);
  }

  // Interpolate between chain-start tangent and current endTangent
  let chainStartTangent: vec3<f32> = data.chain.xyz;
  var tangent: vec3<f32>;
  let cstLen: f32 = length(chainStartTangent);
  if (isArc) {
    // For arcs, use the curve's actual tangent directly (finite difference)
    // to avoid the chain blend suppressing the curve's direction change
    tangent = endTangent;
  } else if (cstLen > 0.001) {
    tangent = normalize(mix(chainStartTangent, endTangent, t));
  } else {
    tangent = endTangent;
  }

  let upDir: vec3<f32> = vec3<f32>(0.0, 0.0, 1.0);
  var rightDir: vec3<f32> = -normalize(cross(upDir, tangent));
  if (length(rightDir) < 0.001) {
    rightDir = vec3<f32>(1.0, 0.0, 0.0);
  }
  let fwdDir: vec3<f32> = -normalize(cross(rightDir, upDir));
  let rot = mat3x3<f32>(rightDir, upDir, fwdDir);

  let hScale: f32 = 1.25;
  let areaCorrection: f32 = 1.1;
  let local: vec3<f32> = vec3<f32>(
    in.position.x * width * areaCorrection,
    in.position.y * width * hScale,
    0.0
  );

  let worldPos: vec3<f32> = segPos + rot * local;
  let worldNormal: vec3<f32> = rot * in.normal;

  var out: VertexOutput;
  out.clipPos = camera.viewProj * vec4<f32>(worldPos, 1.0);
  out.worldPos = worldPos;
  out.worldNormal = worldNormal;
  out.color = segColor * material.baseColorTint;
  return out;
}

@vertex
fn vs_cap(in: VertexInput, @builtin(instance_index) ii: u32) -> VertexOutput {
  let capInfo: vec2<f32> = capInstances[ii];
  let segIdx: u32 = u32(capInfo.x);
  let isEnd: f32 = capInfo.y;

  let data = segments[segIdx];
  let segColor = colors[segIdx].rgb;
  let capsWidth: f32 = data.startPos.w;

  // Position at segment start or end
  let pos: vec3<f32> = select(data.startPos.xyz, data.endPos.xyz, isEnd > 0.5);

  // Tangent direction (facing outward from the endpoint)
  let dir: vec3<f32> = data.endPos.xyz - data.startPos.xyz;
  let segLen: f32 = length(dir);
  var tangent: vec3<f32>;
  if (segLen > 0.001) {
    tangent = dir / segLen;
  } else {
    tangent = vec3<f32>(0.0, 0.0, 1.0);
  }
  // For end caps the dome faces along +tangent, for start caps faces opposite
  let capDir: vec3<f32> = select(-tangent, tangent, isEnd > 0.5);

  // Build orthonormal basis
  let upDir: vec3<f32> = vec3<f32>(0.0, 0.0, 1.0);
  var rightDir: vec3<f32> = -normalize(cross(upDir, capDir));
  if (length(rightDir) < 0.001) {
    rightDir = vec3<f32>(1.0, 0.0, 0.0);
  }
  let fwdDir: vec3<f32> = -normalize(cross(rightDir, upDir));
  let rot = mat3x3<f32>(rightDir, upDir, fwdDir);

  // The cap geometry sits at z=0 (rim) to z=1 (apex).
  // Scale by width.
  let hScale: f32 = 1.25;
  let areaCorrection: f32 = 1.1;
  let local: vec3<f32> = vec3<f32>(
    in.position.x * capsWidth * areaCorrection,
    in.position.y * capsWidth * hScale,
    in.position.z * capsWidth * 0.5
  );

  let worldPos: vec3<f32> = pos + rot * local;
  let worldNormal: vec3<f32> = rot * in.normal;

  var out: VertexOutput;
  out.clipPos = camera.viewProj * vec4<f32>(worldPos, 1.0);
  out.worldPos = worldPos;
  out.worldNormal = worldNormal;
  out.color = segColor * material.baseColorTint;
  return out;
}

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
  let lit: vec3<f32> = in.color * (material.ambientStrength + (diffuse + specular) * NdotL * lightIntensity);
  return vec4<f32>(lit, 1.0);
}
`;

export interface MaterialUniforms {
  roughness: number;
  metalness: number;
  envIntensity: number;
  specularStrength: number;
  ambientStrength: number;
  baseColorTint: [number, number, number];
}

export class SlicedPipeline {
  device: GPUDevice;
  pipeline!: GPURenderPipeline;

  // Buffers
  vertexBuffer!: GPUBuffer;
  indexBuffer!: GPUBuffer;
  indexCount = 0;
  bindGroup!: GPUBindGroup;

  // Uniform buffers
  cameraBuf!: GPUBuffer;
  materialBuf!: GPUBuffer;
  lightDirBuf!: GPUBuffer;
  segmentBuffers!: GpuSegmentBuffers;

  // Bind group layout
  bindGroupLayout!: GPUBindGroupLayout;
  pipelineLayout!: GPUPipelineLayout;

  // Cap geometry buffers
  capVertexBuffer!: GPUBuffer;
  capIndexBuffer!: GPUBuffer;
  capIndexCount = 0;
  capPipeline!: GPURenderPipeline;

  // Current material (for fast UBO writes)
  material: MaterialUniforms = {
    roughness: 0.10,
    metalness: 0.0,
    envIntensity: 0.25,
    specularStrength: 1.0,
    ambientStrength: 0.5,
    baseColorTint: [1.0, 0.878, 0.831],
  };

  lightDir: [number, number, number, number] = [0.416, -0.25, 0.872, 1.0];

  constructor(device: GPUDevice) {
    this.device = device;
  }

  async init() {
    const d = this.device;

    // ── Vertex geometry ──
    const geo = generateBodyGeometry();

    this.vertexBuffer = d.createBuffer({
      size: geo.interleaved.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(this.vertexBuffer.getMappedRange()).set(geo.interleaved);
    this.vertexBuffer.unmap();

    this.indexBuffer = d.createBuffer({
      size: geo.indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Uint16Array(this.indexBuffer.getMappedRange()).set(geo.indices);
    this.indexBuffer.unmap();
    this.indexCount = geo.indices.length;

    // ── Cap geometry ──
    const capGeo = generateCapGeometry();
    this.capVertexBuffer = d.createBuffer({
      size: capGeo.interleaved.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(this.capVertexBuffer.getMappedRange()).set(capGeo.interleaved);
    this.capVertexBuffer.unmap();

    this.capIndexBuffer = d.createBuffer({
      size: capGeo.indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Uint16Array(this.capIndexBuffer.getMappedRange()).set(capGeo.indices);
    this.capIndexBuffer.unmap();
    this.capIndexCount = capGeo.indices.length;

    // ── Bind group layout ──
    this.bindGroupLayout = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }, // camera
        { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }, // material
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }, // segments
        { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }, // colors
        { binding: 4, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }, // lightDir
        { binding: 5, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }, // cap instances
      ],
    });

    this.pipelineLayout = d.createPipelineLayout({
      bindGroupLayouts: [this.bindGroupLayout],
    });

    // ── Uniform buffers ──
    this.cameraBuf = d.createBuffer({
      size: 64 + 16, // mat4x4<f32> (64) + vec3<f32> padded to 16
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.materialBuf = d.createBuffer({
      size: 48, // 5 floats + vec3 (padded correctly)
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.writeMaterialUBO();

    this.lightDirBuf = d.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.writeLightDirUBO();

    // ── Render pipeline ──
    const shaderModule = d.createShaderModule({ code: SHADER_SRC });

    this.pipeline = d.createRenderPipeline({
      layout: this.pipelineLayout,
      vertex: {
        module: shaderModule,
        entryPoint: 'vs_main',
        buffers: [{
          arrayStride: 24, // 6 floats × 4 bytes
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x3' }, // position
            { shaderLocation: 1, offset: 12, format: 'float32x3' }, // normal
          ],
        }],
      },
      fragment: {
        module: shaderModule,
        entryPoint: 'fs_main',
        targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }],
      },
      primitive: {
        topology: 'triangle-list',
        cullMode: 'none',
      },
      depthStencil: {
        format: 'depth24plus',
        depthWriteEnabled: true,
        depthCompare: 'less',
      },
    });

    // ── Cap pipeline ──
    this.capPipeline = d.createRenderPipeline({
      layout: this.pipelineLayout,
      vertex: {
        module: shaderModule,
        entryPoint: 'vs_cap',
        buffers: [{
          arrayStride: 24,
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x3' },
            { shaderLocation: 1, offset: 12, format: 'float32x3' },
          ],
        }],
      },
      fragment: {
        module: shaderModule,
        entryPoint: 'fs_main',
        targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }],
      },
      primitive: {
        topology: 'triangle-list',
        cullMode: 'none',
      },
      depthStencil: {
        format: 'depth24plus',
        depthWriteEnabled: true,
        depthCompare: 'less',
      },
    });
  }

  /** Set the segment data buffers and rebuild the bind group. */
  setSegments(buffers: GpuSegmentBuffers) {
    this.segmentBuffers = buffers;
    this.bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.cameraBuf } },
        { binding: 1, resource: { buffer: this.materialBuf } },
        { binding: 2, resource: { buffer: buffers.segmentBuffer } },
        { binding: 3, resource: { buffer: buffers.colorBuffer } },
        { binding: 4, resource: { buffer: this.lightDirBuf } },
        { binding: 5, resource: { buffer: buffers.capBuffer } },
      ],
    });
  }

  /** Write material uniforms to the GPU buffer. */
  writeMaterialUBO() {
    const m = this.material;
    const data = new Float32Array([
      m.roughness, m.metalness, m.envIntensity, m.specularStrength, m.ambientStrength,
      0, 0, 0, // padding after first 5 floats (fill to 32 bytes)
      m.baseColorTint[0], m.baseColorTint[1], m.baseColorTint[2],
      0, // padding
    ]);
    this.device.queue.writeBuffer(this.materialBuf, 0, data);
  }

  /** Write light direction UBO. */
  writeLightDirUBO() {
    this.device.queue.writeBuffer(this.lightDirBuf, 0, new Float32Array(this.lightDir));
  }

  /** Write camera UBO for the current frame. */
  writeCameraUBO(camera: OrbitCamera) {
    // Structure: mat4x4<f32> (64 bytes) + vec3<f32> (12 bytes) padded to vec4 (16 bytes)
    const data = new Float32Array(20); // 16 for mat4 + 4 for camPos
    data.set(camera.viewProj, 0);
    data[16] = camera.position[0];
    data[17] = camera.position[1];
    data[18] = camera.position[2];
    this.device.queue.writeBuffer(this.cameraBuf, 0, data);
  }

  /** Issue body draw call. Must be inside a render pass. */
  draw(pass: GPURenderPassEncoder) {
    if (!this.segmentBuffers) return;
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.setVertexBuffer(0, this.vertexBuffer);
    pass.setIndexBuffer(this.indexBuffer, 'uint16');
    pass.drawIndexed(this.indexCount, this.segmentBuffers.count);
  }

  /** Issue cap draw call. Must be inside a render pass. */
  drawCaps(pass: GPURenderPassEncoder) {
    if (!this.segmentBuffers || !this.segmentBuffers.capCount) return;
    pass.setPipeline(this.capPipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.setVertexBuffer(0, this.capVertexBuffer);
    pass.setIndexBuffer(this.capIndexBuffer, 'uint16');
    pass.drawIndexed(this.capIndexCount, this.segmentBuffers.capCount);
  }

  /** Dispose all GPU resources. */
  dispose() {
    this.vertexBuffer?.destroy();
    this.indexBuffer?.destroy();
    this.capVertexBuffer?.destroy();
    this.capIndexBuffer?.destroy();
    this.cameraBuf?.destroy();
    this.materialBuf?.destroy();
    this.lightDirBuf?.destroy();
    this.segmentBuffers?.segmentBuffer?.destroy();
    this.segmentBuffers?.colorBuffer?.destroy();
    this.segmentBuffers?.capBuffer?.destroy();
  }
}
