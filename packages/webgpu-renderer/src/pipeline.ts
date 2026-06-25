import { generateAllBodyGeometries, generateAllCapGeometries } from './geometry';
import type { GpuSegmentBuffers } from './buffer';
import { OrbitCamera } from './camera';
import typesWgsl from './shaders/types.wgsl?raw';
import segmentWgsl from './shaders/segment.wgsl?raw';
import capWgsl from './shaders/cap.wgsl?raw';
import pbrWgsl from './shaders/pbr.wgsl?raw';
import cullWgsl from './shaders/cull.wgsl?raw';
import ssaoWgsl from './shaders/ssao.wgsl?raw';
import blurWgsl from './shaders/blur.wgsl?raw';
import shadowWgsl from './shaders/shadow.wgsl?raw';

const SHADER_SRC = typesWgsl + segmentWgsl + capWgsl + pbrWgsl;

export interface MaterialUniforms {
  roughness: number; metalness: number; envIntensity: number;
  specularStrength: number; ambientStrength: number;
  baseColorTint: [number, number, number];
}

export class SlicedPipeline {
  device: GPUDevice;

  // LOD geometry
  bodyVB: GPUBuffer[] = []; bodyIB: GPUBuffer[] = []; bodyIC: number[] = [];
  capVB: GPUBuffer[] = []; capIB: GPUBuffer[] = []; capIC: number[] = [];

  // Render pipelines (3 body, 2 caps)
  bodyPipes: GPURenderPipeline[] = [];
  capPipes: GPURenderPipeline[] = [];
  renderBGL!: GPUBindGroupLayout;
  renderBG!: GPUBindGroup;
  lodBGs: GPUBindGroup[] = [];

  // Legacy refs (LOD 0)
  vertexBuffer!: GPUBuffer; indexBuffer!: GPUBuffer; indexCount = 0;
  capVertexBuffer!: GPUBuffer; capIndexBuffer!: GPUBuffer; capIndexCount = 0;
  pipeline!: GPURenderPipeline; capPipeline!: GPURenderPipeline;

  // Uniform buffers
  cameraBuf!: GPUBuffer; materialBuf!: GPUBuffer; lightDirBuf!: GPUBuffer;
  segmentBuffers!: GpuSegmentBuffers;

  // Compute (LOD culling)
  computePipe!: GPUComputePipeline;
  arcFixupPipe!: GPUComputePipeline;
  computeBGL!: GPUBindGroupLayout;
  computeBG!: GPUBindGroup;
  indirectBuf!: GPUBuffer;
  segmentLodBuf!: GPUBuffer;
  lodCountersBuf!: GPUBuffer;
  cullParamsBuf!: GPUBuffer;
  lodLevelBufs: GPUBuffer[] = [];

  // SSAO textures
  offscreenColorTex!: GPUTexture;
  normalTex!: GPUTexture;
  ssaoDepthTex!: GPUTexture;
  ssaoOcclusionTex!: GPUTexture;
  ssaoWidth = 0; ssaoHeight = 0;

  // SSAO render pass
  ssaoMod!: GPUShaderModule;
  ssaoBGL!: GPUBindGroupLayout;
  ssaoBG!: GPUBindGroup;
  ssaoParamsBuf!: GPUBuffer;
  screenSizeBuf!: GPUBuffer;
  ssaoPipe!: GPURenderPipeline;
  // Composite pipeline (fullscreen quad)
  compositeBGL!: GPUBindGroupLayout;
  compositeBG!: GPUBindGroup;
  compositePipe!: GPURenderPipeline;
  offscreenFormat: GPUTextureFormat = 'rgba8unorm';

  // Bilateral blur (SSAO smoothing)
  blurMod!: GPUShaderModule;
  blurTempTex!: GPUTexture;
  blurBGL!: GPUBindGroupLayout;
  blurParamsBuf!: GPUBuffer;
  blurNearFarBuf!: GPUBuffer;
  blurPipe!: GPURenderPipeline;
  blurHorizBG!: GPUBindGroup;  // reads occlusion, writes temp
  blurVertBG!: GPUBindGroup;   // reads temp, writes back

  // Shadow mapping
  shadowTex!: GPUTexture;
  shadowVPBuf!: GPUBuffer;
  shadowSampler!: GPUSampler;
  shadowBGL!: GPUBindGroupLayout;
  shadowRenderBGL!: GPUBindGroupLayout;
  shadowBG!: GPUBindGroup;
  shadowPassBG!: GPUBindGroup;
  shadowPipe!: GPURenderPipeline;
  shadowMod!: GPUShaderModule;

  // Debug texture preview
  debugBGL!: GPUBindGroupLayout;
  debugDepthBGL!: GPUBindGroupLayout;
  debugPipe!: GPURenderPipeline;
  debugDepthPipe!: GPURenderPipeline;

  material: MaterialUniforms = {
    roughness: 0.10, metalness: 0, envIntensity: 0.25,
    specularStrength: 1, ambientStrength: 0.5,
    baseColorTint: [1, 0.878, 0.831],
  };
  lightDir: [number, number, number, number] = [0.416, -0.25, 0.872, 1];

  constructor(device: GPUDevice) { this.device = device; }

  async init() {
    const d = this.device;

    // ── Geometry ──
    const bodyGeos = generateAllBodyGeometries();
    const capGeos = generateAllCapGeometries();
    const mkVB = (g: { interleaved: Float32Array }) => { const b = d.createBuffer({ size: g.interleaved.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST, mappedAtCreation: true }); new Float32Array(b.getMappedRange()).set(g.interleaved); b.unmap(); return b; };
    const mkIB = (g: { indices: Uint16Array }) => { const b = d.createBuffer({ size: g.indices.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST, mappedAtCreation: true }); new Uint16Array(b.getMappedRange()).set(g.indices); b.unmap(); return b; };
    this.bodyVB = bodyGeos.map(mkVB); this.bodyIB = bodyGeos.map(mkIB); this.bodyIC = bodyGeos.map(g => g.indices.length);
    this.capVB = capGeos.map(mkVB); this.capIB = capGeos.map(mkIB); this.capIC = capGeos.map(g => g.indices.length);
    [this.vertexBuffer, this.indexBuffer, this.indexCount] = [this.bodyVB[0], this.bodyIB[0], this.bodyIC[0]];
    [this.capVertexBuffer, this.capIndexBuffer, this.capIndexCount] = [this.capVB[0], this.capIB[0], this.capIC[0]];

    // ── Shader modules ──
    const shaderMod = d.createShaderModule({ code: SHADER_SRC });
    const cullMod = d.createShaderModule({ code: cullWgsl });

    // ── Bind group layouts ──
    this.renderBGL = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 4, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 5, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 6, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 7, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
      ],
    });
    this.computeBGL = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });

    // ── Uniform buffers ──
    this.cameraBuf = d.createBuffer({ size: 144, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.materialBuf = d.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.lightDirBuf = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.writeMaterialUBO(); this.writeLightDirUBO();

    // ── Culling buffers ──
    this.indirectBuf = d.createBuffer({
      size: 3 * 20,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT,
    });
    this.lodCountersBuf = d.createBuffer({
      size: 3 * 4, // 3 × u32
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.cullParamsBuf = d.createBuffer({
      size: 8,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.lodLevelBufs = [0, 1, 2].map(lod => {
      const b = d.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
      new Uint32Array(b.getMappedRange()).set([lod]);
      b.unmap();
      return b;
    });

    // ── SSAO render pass ──
    this.ssaoMod = d.createShaderModule({ code: ssaoWgsl });
    this.ssaoParamsBuf = d.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.screenSizeBuf = d.createBuffer({ size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(this.ssaoParamsBuf, 0, new Float32Array([0.25, 0.25, 0.01, 1.5, 0, 0, 0, 0]));

    // SSAO bind group layout (group 0): depth + params + screenSize
    this.ssaoBGL = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      ],
    });

    // Composite bind group layout (group 1): offscreenColor + occlusion
    this.compositeBGL = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
      ],
    });

    // SSAO render pipeline: fullscreen quad → occlusion texture (r32float)
    const ssaoPL = d.createPipelineLayout({ bindGroupLayouts: [this.ssaoBGL] });
    this.ssaoPipe = d.createRenderPipeline({
      layout: ssaoPL,
      vertex: { module: this.ssaoMod, entryPoint: 'vs_fullscreen' },
      fragment: { module: this.ssaoMod, entryPoint: 'fs_ssao', targets: [{ format: 'r32float' }] },
      primitive: { topology: 'triangle-list' },
    });

    // ── Shadow mapping ──
    this.shadowMod = d.createShaderModule({ code: shadowWgsl });
    this.shadowVPBuf = d.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.shadowSampler = d.createSampler({
      compare: 'less',
      magFilter: 'linear',
      minFilter: 'linear',
    });
    // Shadow BGL (group 0: lightVP + segments for shadow render pass)
    this.shadowBGL = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      ],
    });
    // Shadow render BGL (used at group 1 of main render pass)
    this.shadowRenderBGL = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
      ],
    });

    // ── Render pipelines ──
    const vx = { arrayStride: 24, attributes: [
      { shaderLocation: 0, offset: 0, format: 'float32x3' as const },
      { shaderLocation: 1, offset: 12, format: 'float32x3' as const },
    ]};
    const ds = { format: 'depth32float' as const, depthWriteEnabled: true, depthCompare: 'less' as const };
    const fmt = navigator.gpu.getPreferredCanvasFormat();
    this.offscreenFormat = fmt;
    const rPL = d.createPipelineLayout({ bindGroupLayouts: [this.renderBGL, this.shadowRenderBGL] });

    for (let l = 0; l < 3; l++) this.bodyPipes.push(d.createRenderPipeline({
      layout: rPL, vertex: { module: shaderMod, entryPoint: 'vs_main', buffers: [vx] },
      fragment: { module: shaderMod, entryPoint: 'fs_main', targets: [{ format: fmt }, { format: 'rgba8unorm' }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' }, depthStencil: ds,
    }));
    for (let l = 0; l < 2; l++) this.capPipes.push(d.createRenderPipeline({
      layout: rPL, vertex: { module: shaderMod, entryPoint: 'vs_cap', buffers: [vx] },
      fragment: { module: shaderMod, entryPoint: 'fs_main', targets: [{ format: fmt }, { format: 'rgba8unorm' }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' }, depthStencil: ds,
    }));
    this.pipeline = this.bodyPipes[0]; this.capPipeline = this.capPipes[0];

    // ── Shadow pipeline (depth-only) ──
    this.shadowTex = d.createTexture({
      size: [2048, 2048], format: 'depth32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    const shadowPL = d.createPipelineLayout({ bindGroupLayouts: [this.shadowBGL] });
    this.shadowPipe = d.createRenderPipeline({
      layout: shadowPL,
      vertex: { module: this.shadowMod, entryPoint: 'vs_shadow', buffers: [vx] },
      fragment: { module: this.shadowMod, entryPoint: 'fs_shadow', targets: [] },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
    });

    // ── Composite pipeline (fullscreen quad) ──
    const compPL = d.createPipelineLayout({ bindGroupLayouts: [null, this.compositeBGL] });
    this.compositePipe = d.createRenderPipeline({
      layout: compPL,
      vertex: { module: this.ssaoMod, entryPoint: 'vs_fullscreen' },
      fragment: { module: this.ssaoMod, entryPoint: 'fs_composite', targets: [{ format: fmt }] },
      primitive: { topology: 'triangle-list' },
    });

    // ── Debug preview pipelines ──
    // Float textures (offscreen color, occlusion)
    this.debugBGL = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
      ],
    });
    const debugPL = d.createPipelineLayout({ bindGroupLayouts: [this.debugBGL] });
    this.debugPipe = d.createRenderPipeline({
      layout: debugPL,
      vertex: { module: this.ssaoMod, entryPoint: 'vs_fullscreen' },
      fragment: { module: this.ssaoMod, entryPoint: 'fs_debug', targets: [{ format: fmt }] },
      primitive: { topology: 'triangle-list' },
    });
    // Depth textures (SSAO depth, shadow map)
    this.debugDepthBGL = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
      ],
    });
    const debugDepthPL = d.createPipelineLayout({ bindGroupLayouts: [this.debugDepthBGL] });
    this.debugDepthPipe = d.createRenderPipeline({
      layout: debugDepthPL,
      vertex: { module: this.ssaoMod, entryPoint: 'vs_fullscreen' },
      fragment: { module: this.ssaoMod, entryPoint: 'fs_debug_depth', targets: [{ format: fmt }] },
      primitive: { topology: 'triangle-list' },
    });

    // ── Bilateral blur pipeline ──
    this.blurMod = d.createShaderModule({ code: blurWgsl });
    this.blurBGL = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });
    this.blurParamsBuf = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.blurNearFarBuf = d.createBuffer({ size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const blurPL = d.createPipelineLayout({ bindGroupLayouts: [this.blurBGL] });
    this.blurPipe = d.createRenderPipeline({
      layout: blurPL,
      vertex: { module: this.blurMod, entryPoint: 'vs_fullscreen' },
      fragment: { module: this.blurMod, entryPoint: 'fs_blur', targets: [{ format: 'r32float' }] },
      primitive: { topology: 'triangle-list' },
    });

    // ── Compute pipeline ──
    const cPL = d.createPipelineLayout({ bindGroupLayouts: [this.computeBGL] });
    this.computePipe = d.createComputePipeline({
      layout: cPL,
      compute: { module: cullMod, entryPoint: 'cs_main' },
    });
    this.arcFixupPipe = d.createComputePipeline({
      layout: cPL,
      compute: { module: cullMod, entryPoint: 'cs_arc_fixup' },
    });
  }

  setSegments(buffers: GpuSegmentBuffers) {
    this.segmentBuffers = buffers;

    // Segment LOD buffer (atomic u32 per segment)
    if (this.segmentLodBuf) this.segmentLodBuf.destroy();
    this.segmentLodBuf = this.device.createBuffer({
      size: buffers.count * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Render bind group
    this.renderBG = this.device.createBindGroup({
      layout: this.renderBGL,
      entries: [
        { binding: 0, resource: { buffer: this.cameraBuf } },
        { binding: 1, resource: { buffer: this.materialBuf } },
        { binding: 2, resource: { buffer: buffers.segmentBuffer } },
        { binding: 3, resource: { buffer: buffers.colorBuffer } },
        { binding: 4, resource: { buffer: this.lightDirBuf } },
        { binding: 5, resource: { buffer: buffers.capBuffer } },
        { binding: 6, resource: { buffer: this.segmentLodBuf } },
        { binding: 7, resource: { buffer: this.lodLevelBufs[0] } },
      ],
    });

    // Per-LOD bind groups
    this.lodBGs = [0, 1, 2].map(lod => this.device.createBindGroup({
      layout: this.renderBGL,
      entries: [
        { binding: 0, resource: { buffer: this.cameraBuf } },
        { binding: 1, resource: { buffer: this.materialBuf } },
        { binding: 2, resource: { buffer: buffers.segmentBuffer } },
        { binding: 3, resource: { buffer: buffers.colorBuffer } },
        { binding: 4, resource: { buffer: this.lightDirBuf } },
        { binding: 5, resource: { buffer: buffers.capBuffer } },
        { binding: 6, resource: { buffer: this.segmentLodBuf } },
        { binding: 7, resource: { buffer: this.lodLevelBufs[lod] } },
      ],
    }));

    // Compute bind group (segments + camera + indirect + lod + cullParams)
    this.computeBG = this.device.createBindGroup({
      layout: this.computeBGL,
      entries: [
        { binding: 0, resource: { buffer: buffers.segmentBuffer } },
        { binding: 1, resource: { buffer: this.cameraBuf } },
        { binding: 2, resource: { buffer: this.indirectBuf } },
        { binding: 3, resource: { buffer: this.segmentLodBuf } },
        { binding: 4, resource: { buffer: this.lodCountersBuf } },
        { binding: 5, resource: { buffer: this.cullParamsBuf } },
      ],
    });

    // Shadow render bind group (main pass, group 1)
    this.shadowBG = this.device.createBindGroup({
      layout: this.shadowRenderBGL,
      entries: [
        { binding: 0, resource: this.shadowTex.createView() },
        { binding: 1, resource: this.shadowSampler },
        { binding: 2, resource: { buffer: this.shadowVPBuf } },
      ],
    });
    // Shadow pass bind group (depth-only, group 0)
    this.shadowPassBG = this.device.createBindGroup({
      layout: this.shadowBGL,
      entries: [
        { binding: 0, resource: { buffer: this.shadowVPBuf } },
        { binding: 1, resource: { buffer: buffers.segmentBuffer } },
      ],
    });
  }

  /** Zero out indirect draw instance counts and LOD counters. */
  resetIndirect() {
    const a = new Uint32Array(15);
    for (let l = 0; l < 3; l++) {
      a[l * 5 + 0] = this.bodyIC[l];
      a[l * 5 + 1] = 0;
    }
    this.device.queue.writeBuffer(this.indirectBuf, 0, a);
    this.device.queue.writeBuffer(this.lodCountersBuf, 0, new Uint32Array(3));
  }

  /** Dispatch LOD culling, then copy counters to indirect buffer. */
  dispatchCull(vpHeight: number) {
    if (!this.segmentBuffers) return;
    const count = this.segmentBuffers.count;

    this.device.queue.writeBuffer(this.cullParamsBuf, 0,
      new Float32Array([vpHeight, Math.tan(Math.PI / 6)]));

    const enc = this.device.createCommandEncoder();

    // Pass 1: evaluate LOD for all segments
    const pass1 = enc.beginComputePass();
    pass1.setPipeline(this.computePipe);
    pass1.setBindGroup(0, this.computeBG);
    pass1.dispatchWorkgroups(Math.ceil(count / 64));
    pass1.end();

    // Pass 2: arcs inherit LOD from predecessor
    const pass2 = enc.beginComputePass();
    pass2.setPipeline(this.arcFixupPipe);
    pass2.setBindGroup(0, this.computeBG);
    pass2.dispatchWorkgroups(Math.ceil(count / 64));
    pass2.end();

    // Copy LOD counters into indirect draw buffer
    for (let l = 0; l < 3; l++) {
      enc.copyBufferToBuffer(this.lodCountersBuf, l * 4, this.indirectBuf, l * 20 + 4, 4);
    }

    this.device.queue.submit([enc.finish()]);
  }

  /** Draw body for each LOD. Each draws all instances — vertex shader culls by LOD. */
  drawBody(pass: GPURenderPassEncoder) {
    if (!this.segmentBuffers) return;
    for (let l = 0; l < 3; l++) {
      pass.setPipeline(this.bodyPipes[l]);
      pass.setBindGroup(0, this.lodBGs[l]);
      pass.setBindGroup(1, this.shadowBG);
      pass.setVertexBuffer(0, this.bodyVB[l]);
      pass.setIndexBuffer(this.bodyIB[l], 'uint16');
      pass.drawIndexed(this.bodyIC[l], this.segmentBuffers.count);
    }
  }

  /** Draw caps (LOD 0 only, all instances). */
  drawCaps(pass: GPURenderPassEncoder) {
    if (!this.segmentBuffers || !this.segmentBuffers.capCount) return;
    pass.setPipeline(this.capPipes[0]);
    pass.setBindGroup(0, this.lodBGs[0]);
    pass.setBindGroup(1, this.shadowBG);
    pass.setVertexBuffer(0, this.capVB[0]);
    pass.setIndexBuffer(this.capIB[0], 'uint16');
    pass.drawIndexed(this.capIC[0], this.segmentBuffers.capCount);
  }

  writeMaterialUBO() {
    const m = this.material;
    this.device.queue.writeBuffer(this.materialBuf, 0, new Float32Array([
      m.roughness, m.metalness, m.envIntensity, m.specularStrength, m.ambientStrength,
      0, 0, 0,
      m.baseColorTint[0], m.baseColorTint[1], m.baseColorTint[2], 0,
    ]));
  }
  writeLightDirUBO() { this.device.queue.writeBuffer(this.lightDirBuf, 0, new Float32Array(this.lightDir)); }
  writeCameraUBO(camera: OrbitCamera) {
    const d = new Float32Array(36);
    d.set(camera.viewProj, 0);
    d.set(camera.viewMat, 16);
    d[32] = camera.position[0]; d[33] = camera.position[1]; d[34] = camera.position[2];
    this.device.queue.writeBuffer(this.cameraBuf, 0, d);
  }

  /** Write camera near/far/fov into the SSAO params buffer for depth linearization. */
  setSSAOCamera(near: number, far: number, fov: number) {
    const fovScale = Math.tan(fov / 2) * 2;
    this.device.queue.writeBuffer(this.ssaoParamsBuf, 16, new Float32Array([near, far, fovScale, 0]));
    this.device.queue.writeBuffer(this.blurNearFarBuf, 0, new Float32Array([near, far]));
  }

  /** Create or recreate offscreen textures + bind groups at given resolution. */
  resizeSSAO(w: number, h: number) {
    if (w === this.ssaoWidth && h === this.ssaoHeight) return;
    const d = this.device;
    // Destroy old textures
    for (const t of [this.offscreenColorTex, this.normalTex, this.ssaoDepthTex, this.ssaoOcclusionTex, this.blurTempTex]) t?.destroy();

    this.offscreenColorTex = d.createTexture({
      size: [w, h], format: this.offscreenFormat, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.normalTex = d.createTexture({
      size: [w, h], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.ssaoDepthTex = d.createTexture({
      size: [w, h], format: 'depth32float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.ssaoOcclusionTex = d.createTexture({
      size: [w, h], format: 'r32float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.blurTempTex = d.createTexture({
      size: [w, h], format: 'r32float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });

    // Write screen size uniform
    d.queue.writeBuffer(this.screenSizeBuf, 0, new Float32Array([w, h]));

    // SSAO bind group (group 0): depth texture + params + screenSize
    this.ssaoBG = d.createBindGroup({
      layout: this.ssaoBGL,
      entries: [
        { binding: 0, resource: this.ssaoDepthTex.createView() },
        { binding: 1, resource: { buffer: this.ssaoParamsBuf } },
        { binding: 2, resource: { buffer: this.screenSizeBuf } },
        { binding: 3, resource: this.normalTex.createView() },
      ],
    });

    // Composite bind group (group 1): offscreen color + occlusion
    this.compositeBG = d.createBindGroup({
      layout: this.compositeBGL,
      entries: [
        { binding: 0, resource: this.offscreenColorTex.createView() },
        { binding: 1, resource: this.ssaoOcclusionTex.createView() },
      ],
    });

    this.ssaoWidth = w;
    this.ssaoHeight = h;

    // Blur bind groups: horizontal (occlusion→temp) and vertical (temp→occlusion)
    this.blurHorizBG = d.createBindGroup({
      layout: this.blurBGL,
      entries: [
        { binding: 0, resource: this.ssaoOcclusionTex.createView() },
        { binding: 1, resource: this.ssaoDepthTex.createView() },
        { binding: 2, resource: { buffer: this.blurParamsBuf } },
        { binding: 3, resource: { buffer: this.blurNearFarBuf } },
      ],
    });
    this.blurVertBG = d.createBindGroup({
      layout: this.blurBGL,
      entries: [
        { binding: 0, resource: this.blurTempTex.createView() },
        { binding: 1, resource: this.ssaoDepthTex.createView() },
        { binding: 2, resource: { buffer: this.blurParamsBuf } },
        { binding: 3, resource: { buffer: this.blurNearFarBuf } },
      ],
    });
  }

  /** Render SSAO to occlusion texture. */
  dispatchSSAO(enc: GPUCommandEncoder) {
    if (!this.ssaoWidth || !this.ssaoHeight) return;
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view: this.ssaoOcclusionTex.createView(),
        loadOp: 'clear',
        storeOp: 'store',
      }],
    });
    pass.setPipeline(this.ssaoPipe);
    pass.setBindGroup(0, this.ssaoBG);
    pass.draw(3);
    pass.end();
  }

  /** Bilateral blur the occlusion texture (2-pass separable). */
  dispatchBlur(enc: GPUCommandEncoder) {
    if (!this.ssaoWidth || !this.ssaoHeight) return;
    const d = this.device;

    // Horizontal pass: read occlusion → write temp
    d.queue.writeBuffer(this.blurParamsBuf, 0, new Float32Array([1, 0, this.ssaoWidth, this.ssaoHeight]));
    {
      const pass = enc.beginRenderPass({
        colorAttachments: [{
          view: this.blurTempTex.createView(),
          loadOp: 'clear',
          storeOp: 'store',
        }],
      });
      pass.setPipeline(this.blurPipe);
      pass.setBindGroup(0, this.blurHorizBG);
      pass.draw(3);
      pass.end();
    }

    // Vertical pass: read temp → write back to occlusion
    d.queue.writeBuffer(this.blurParamsBuf, 0, new Float32Array([0, 1, this.ssaoWidth, this.ssaoHeight]));
    {
      const pass = enc.beginRenderPass({
        colorAttachments: [{
          view: this.ssaoOcclusionTex.createView(),
          loadOp: 'clear',
          storeOp: 'store',
        }],
      });
      pass.setPipeline(this.blurPipe);
      pass.setBindGroup(0, this.blurVertBG);
      pass.draw(3);
      pass.end();
    }
  }  /** Render shadow map (depth-only from light POV). */
  renderShadowMap(enc: GPUCommandEncoder) {
    if (!this.segmentBuffers) return;
    enc.pushDebugGroup('shadow-map');
    const pass = enc.beginRenderPass({
      colorAttachments: [],
      depthStencilAttachment: {
        view: this.shadowTex.createView(),
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    });
    pass.setPipeline(this.shadowPipe);
    pass.setBindGroup(0, this.shadowPassBG);
    // Draw all LOD 0 body geometry (simplest = enough for shadows)
    pass.setVertexBuffer(0, this.bodyVB[1]);
    pass.setIndexBuffer(this.bodyIB[1], 'uint16');
    pass.drawIndexed(this.bodyIC[1], this.segmentBuffers.count);
    pass.end();
    enc.popDebugGroup();
  }

  /** Render a debug preview of one internal texture to a render pass. */
  renderDebugView(pass: GPURenderPassEncoder, mode: string) {
    let view: GPUTextureView;
    let depthMode = false;
    switch (mode) {
      case 'depth':
        view = this.ssaoDepthTex.createView();
        depthMode = true;
        break;
      case 'occlusion':
        view = this.ssaoOcclusionTex.createView();
        break;
      case 'color':
        view = this.offscreenColorTex.createView();
        break;
      case 'shadow':
        view = this.shadowTex.createView();
        depthMode = true;
        break;
      default:
        return;
    }
    const bg = this.device.createBindGroup({
      layout: depthMode ? this.debugDepthBGL : this.debugBGL,
      entries: [{ binding: 0, resource: view }],
    });
    pass.setPipeline(depthMode ? this.debugDepthPipe : this.debugPipe);
    pass.setBindGroup(0, bg);
    pass.draw(3);
  }
  composite(pass: GPURenderPassEncoder) {
    if (!this.compositeBG) return;
    pass.setPipeline(this.compositePipe);
    pass.setBindGroup(1, this.compositeBG);
    pass.draw(3); // fullscreen triangle
  }

  dispose() {
    for (const b of this.bodyVB) b?.destroy();
    for (const b of this.bodyIB) b?.destroy();
    for (const b of this.capVB) b?.destroy();
    for (const b of this.capIB) b?.destroy();
    this.cameraBuf?.destroy(); this.materialBuf?.destroy(); this.lightDirBuf?.destroy();
    this.indirectBuf?.destroy(); this.segmentLodBuf?.destroy(); this.lodCountersBuf?.destroy(); this.cullParamsBuf?.destroy();
    for (const b of this.lodLevelBufs) b?.destroy();
    this.segmentBuffers?.segmentBuffer?.destroy();
    this.segmentBuffers?.colorBuffer?.destroy();
    this.segmentBuffers?.capBuffer?.destroy();
    for (const t of [this.offscreenColorTex, this.normalTex, this.ssaoDepthTex, this.ssaoOcclusionTex, this.blurTempTex]) t?.destroy();
    this.ssaoParamsBuf?.destroy();
    this.shadowTex?.destroy();
    this.shadowVPBuf?.destroy();
  }
}
