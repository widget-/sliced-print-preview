import { generateAllBodyGeometries, generateAllCapGeometries } from './geometry';
import type { GpuSegmentBuffers } from './buffer';
import { OrbitCamera } from './camera';
import typesWgsl from './shaders/types.wgsl?raw';
import segmentWgsl from './shaders/segment.wgsl?raw';
import capWgsl from './shaders/cap.wgsl?raw';
import pbrWgsl from './shaders/pbr.wgsl?raw';

const SHADER_SRC = typesWgsl + segmentWgsl + capWgsl + pbrWgsl;

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

  // LOD geometry arrays
  bodyVertexBuffers: GPUBuffer[] = [];
  bodyIndexBuffers: GPUBuffer[] = [];
  bodyIndexCounts: number[] = [];
  capVertexBuffers: GPUBuffer[] = [];
  capIndexBuffers: GPUBuffer[] = [];
  capIndexCounts: number[] = [];

  // Cap buffers (LOD 0 fallback)
  capVertexBuffer!: GPUBuffer;
  capIndexBuffer!: GPUBuffer;
  capIndexCount = 0;
  capPipeline!: GPURenderPipeline;

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

    // ── Vertex geometry (3 LOD levels) ──
    const bodyGeos = generateAllBodyGeometries();
    const capGeos = generateAllCapGeometries();

    this.bodyVertexBuffers = bodyGeos.map((geo) => {
      const buf = d.createBuffer({
        size: geo.interleaved.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
      });
      new Float32Array(buf.getMappedRange()).set(geo.interleaved);
      buf.unmap();
      return buf;
    });
    this.bodyIndexBuffers = bodyGeos.map((geo) => {
      const buf = d.createBuffer({
        size: geo.indices.byteLength,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
      });
      new Uint16Array(buf.getMappedRange()).set(geo.indices);
      buf.unmap();
      return buf;
    });
    this.bodyIndexCounts = bodyGeos.map((g) => g.indices.length);

    this.capVertexBuffers = capGeos.map((geo) => {
      const buf = d.createBuffer({
        size: geo.interleaved.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
      });
      new Float32Array(buf.getMappedRange()).set(geo.interleaved);
      buf.unmap();
      return buf;
    });
    this.capIndexBuffers = capGeos.map((geo) => {
      const buf = d.createBuffer({
        size: geo.indices.byteLength,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
      });
      new Uint16Array(buf.getMappedRange()).set(geo.indices);
      buf.unmap();
      return buf;
    });
    this.capIndexCounts = capGeos.map((g) => g.indices.length);

    // Use LOD 0 for current rendering (single pipeline mode)
    this.vertexBuffer = this.bodyVertexBuffers[0];
    this.indexBuffer = this.bodyIndexBuffers[0];
    this.indexCount = this.bodyIndexCounts[0];
    this.capVertexBuffer = this.capVertexBuffers[0];
    this.capIndexBuffer = this.capIndexBuffers[0];
    this.capIndexCount = this.capIndexCounts[0];

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
    for (const b of this.bodyVertexBuffers) b?.destroy();
    for (const b of this.bodyIndexBuffers) b?.destroy();
    for (const b of this.capVertexBuffers) b?.destroy();
    for (const b of this.capIndexBuffers) b?.destroy();
    this.cameraBuf?.destroy();
    this.materialBuf?.destroy();
    this.lightDirBuf?.destroy();
    this.segmentBuffers?.segmentBuffer?.destroy();
    this.segmentBuffers?.colorBuffer?.destroy();
    this.segmentBuffers?.capBuffer?.destroy();
  }
}
