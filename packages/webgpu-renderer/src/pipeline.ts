import { generateBodyGeometry } from './geometry';
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
  startPos: vec3<f32>,
  width: f32,
  endPos: vec3<f32>,
  pack0: f32,
  chainTangent: vec3<f32>,
  layerZ: f32,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> material: Material;
@group(0) @binding(2) var<storage, read> segments: array<SegmentData>;
@group(0) @binding(3) var<storage, read> colors: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> lightDir: vec4<f32>;

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
  let segPos: vec3<f32> = mix(data.startPos, data.endPos, t);

  let dir: vec3<f32> = data.endPos - data.startPos;
  let segLen: f32 = length(dir);
  var tangent: vec3<f32>;
  if (segLen > 0.001) {
    tangent = dir / segLen;
  } else {
    tangent = vec3<f32>(0.0, 0.0, 1.0);
  }

  let upDir: vec3<f32> = vec3<f32>(0.0, 0.0, 1.0);
  var rightDir: vec3<f32> = normalize(cross(upDir, tangent));
  if (length(rightDir) < 0.001) {
    rightDir = vec3<f32>(1.0, 0.0, 0.0);
  }
  let fwdDir: vec3<f32> = cross(rightDir, upDir);
  let rot = mat3x3<f32>(rightDir, upDir, fwdDir);

  let hScale: f32 = 1.25;
  let areaCorrection: f32 = 1.1;
  let local: vec3<f32> = vec3<f32>(
    in.position.x * data.width * areaCorrection,
    in.position.y * data.width * hScale,
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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let N: vec3<f32> = normalize(in.worldNormal);
  let L: vec3<f32> = normalize(lightDir.xyz);
  let V: vec3<f32> = normalize(camera.camPos - in.worldPos);
  let H: vec3<f32> = normalize(L + V);

  let NdotL: f32 = max(dot(N, L), 0.0);
  let diffuse: f32 = NdotL * (1.0 - material.metalness);

  let NdotH: f32 = max(dot(N, H), 0.0);
  let roughness2: f32 = material.roughness * material.roughness;
  let specPower: f32 = 2.0 / (roughness2 * roughness2) - 2.0;
  let specular: f32 = pow(NdotH, max(specPower, 1.0)) * material.specularStrength;

  let ambient: f32 = material.ambientStrength;
  let lightIntensity: f32 = lightDir.w;

  let lit: vec3<f32> = in.color * (ambient + diffuse * lightIntensity) + vec3<f32>(specular * lightIntensity);
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

    // ── Bind group layout ──
    this.bindGroupLayout = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }, // camera
        { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }, // material
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }, // segments
        { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }, // colors
        { binding: 4, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }, // lightDir
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

  /** Issue a single draw call. Must be inside a render pass. */
  draw(pass: GPURenderPassEncoder) {
    if (!this.segmentBuffers) return;
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.setVertexBuffer(0, this.vertexBuffer);
    pass.setIndexBuffer(this.indexBuffer, 'uint16');
    pass.drawIndexed(this.indexCount, this.segmentBuffers.count);
  }

  /** Dispose all GPU resources. */
  dispose() {
    this.vertexBuffer?.destroy();
    this.indexBuffer?.destroy();
    this.cameraBuf?.destroy();
    this.materialBuf?.destroy();
    this.lightDirBuf?.destroy();
    this.segmentBuffers?.segmentBuffer?.destroy();
    this.segmentBuffers?.colorBuffer?.destroy();
  }
}
