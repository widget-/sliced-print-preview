import { generateAllBodyGeometries, generateAllCapGeometries } from './geometry';
import type { GpuSegmentBuffers } from './buffer';
import { OrbitCamera } from './camera';
import typesWgsl from './shaders/types.wgsl?raw';
import segmentWgsl from './shaders/segment.wgsl?raw';
import capWgsl from './shaders/cap.wgsl?raw';
import pbrWgsl from './shaders/pbr.wgsl?raw';
import cullWgsl from './shaders/cull.wgsl?raw';
import ssaoWgsl from './shaders/ssao.wgsl?raw';
import ssaoMainWgsl from './shaders/ssao_main.wgsl?raw';
import debugDepthWgsl from './shaders/debug_depth.wgsl?raw';
import blurWgsl from './shaders/blur.wgsl?raw';
import shadowWgsl from './shaders/shadow.wgsl?raw';
import contactShadowWgsl from './shaders/contact_shadow.wgsl?raw';
import velocityWgsl from './shaders/velocity.wgsl?raw';
import taaResolveWgsl from './shaders/taa_resolve.wgsl?raw';
import { IBLPipeline } from './ibl';
import groundWgsl from './shaders/ground.wgsl?raw';

const SHADER_SRC = typesWgsl + segmentWgsl + capWgsl + pbrWgsl;

export interface MaterialUniforms {
  roughness: number; metalness: number; envIntensity: number;
  specularStrength: number; ambientStrength: number;
  arcCurvature: number;
  baseColorTint: [number, number, number];
  useRoleColors?: number;
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
  ssaoKernelBuf!: GPUBuffer;   // 32 × vec4 hemisphere samples (storage)
  ssaoProjBuf!: GPUBuffer;     // projection matrix (uniform)
  ssaoPipe!: GPURenderPipeline;
  // Composite pipeline (fullscreen quad)
  compositeBGL!: GPUBindGroupLayout;
  compositeBG!: GPUBindGroup;
  compositePipe!: GPURenderPipeline;
  // Contact shadow (screen-space ray march)
  contactShadowMod!: GPUShaderModule;
  contactShadowBGL!: GPUBindGroupLayout;
  contactShadowBG!: GPUBindGroup;
  contactShadowPipe!: GPURenderPipeline;
  contactShadowTex!: GPUTexture;
  contactShadowBuf!: GPUBuffer;
  offscreenFormat: GPUTextureFormat = 'rgba8unorm';
  taaFormat: GPUTextureFormat = 'rgba16float';

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
  shadowParamsBuf!: GPUBuffer; // f32 softness
  shadowSampler!: GPUSampler;
  shadowBGL!: GPUBindGroupLayout;
  shadowRenderBGL!: GPUBindGroupLayout;
  shadowBG!: GPUBindGroup;
  shadowPassBG!: GPUBindGroup;
  shadowPipe!: GPURenderPipeline;
  shadowMod!: GPUShaderModule;
  // Second light shadow
  shadowTex2!: GPUTexture;
  shadowVPBuf2!: GPUBuffer;
  shadowRenderBGL2!: GPUBindGroupLayout;
  shadowBG2!: GPUBindGroup;
  shadowPassBG2!: GPUBindGroup;
  shadowPipe2!: GPURenderPipeline;

  // Velocity buffer (for TAA reprojection)
  velocityMod!: GPUShaderModule;
  velocityBGL!: GPUBindGroupLayout;
  velocityPipe!: GPURenderPipeline;
  velocityBuf!: GPUBuffer;
  velocityTex!: GPUTexture;
  velocityBG!: GPUBindGroup;

  // TAA resolve
  taaMod!: GPUShaderModule;
  taaBGL!: GPUBindGroupLayout;
  taaPipe!: GPURenderPipeline;
  taaParamsBuf!: GPUBuffer;
  historyTex: GPUTexture[] = [];
  compositeTex!: GPUTexture;
  taaBG: GPUBindGroup[] = [];
  taaEnabled = true;
  taaFrame = 0;
  _historyIndex = 0;

  // IBL (environment lighting)
  iblPipeline!: IBLPipeline;
  iblBGL!: GPUBindGroupLayout;
  iblBG!: GPUBindGroup;
  _envMapUrl = 'ferndale_studio_07_1k.hdr';

  // Debug texture preview
  debugBGL!: GPUBindGroupLayout;
  debugFloatBGL!: GPUBindGroupLayout;
  debugDepthBGL!: GPUBindGroupLayout;
  debugPipe!: GPURenderPipeline;
  debugFloatPipe!: GPURenderPipeline;
  debugDepthPipe!: GPURenderPipeline;
  debugColorBGL!: GPUBindGroupLayout;
  debugColorPipe!: GPURenderPipeline;
  // Passthrough copy (SSAO-off path)
  copyPipe!: GPURenderPipeline;

  // Ground plane
  groundVB!: GPUBuffer;
  groundIB!: GPUBuffer;
  groundBGL!: GPUBindGroupLayout;
  groundBG!: GPUBindGroup;
  groundPipe!: GPURenderPipeline;
  groundShadowBGL!: GPUBindGroupLayout;
  groundShadowBG!: GPUBindGroup;
  groundShadowPipe!: GPURenderPipeline;

  /** GPU timestamp query set for pass timing (set each frame from main.ts). */
  _gpuQuerySet?: GPUQuerySet;
  /** Mutable query index counter — incremented by each render pass. */
  _gpuQueryIdx = 0;

  material: MaterialUniforms = {
    roughness: 0.65, metalness: 0, envIntensity: 1.0,
    specularStrength: 1, ambientStrength: 0.5,
    arcCurvature: 1,
    baseColorTint: [1, 0.878, 0.831],
    useRoleColors: 1,
  };
  lightDir: [number, number, number, number] = [0.416, -0.25, 0.872, 1];
  lightDir2: [number, number, number, number] = [-0.5, -0.3, 0.6, 0.4];
  lightDir2Buf!: GPUBuffer;
  arcCurvatureBuf!: GPUBuffer;
  /** Shadow PCF kernel radius multiplier (1=1 texel, 2=2 texels, etc). */
  shadowSoftness = 2.0;
  /** Contact shadow ray max distance in world units. */
  contactShadowDist = 0.05;
  /** Contact shadow visibility strength (0=off, 1=full). */
  contactShadowStrength = 0.5;
  /** SSAO sampling radius. */
  ssaoRadius = 0.06;
  /** SSAO occlusion intensity. */
  ssaoIntensity = 3.0;
  /** Stored model bounding box center for shadow VP recomputation. */
  _modelCenter: [number, number, number] = [0, 0, 0];
  _modelExtent: [number, number, number] = [1, 1, 1];

  constructor(device: GPUDevice) { this.device = device; }

  async init() {
    const d = this.device;

    // ── Geometry ──
    const bodyGeos = generateAllBodyGeometries();
    const capGeos = generateAllCapGeometries();
    const mkVB = (g: { interleaved: Float32Array }) => { const b = d.createBuffer({ size: g.interleaved.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST, mappedAtCreation: true }); new Float32Array(b.getMappedRange()).set(g.interleaved); b.unmap(); return b; };
    const mkIB = (g: { indices: Uint16Array }) => { const s = g.indices.byteLength; const a = Math.ceil(s / 4) * 4; const b = d.createBuffer({ size: a, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST, mappedAtCreation: true }); new Uint16Array(b.getMappedRange()).set(g.indices); b.unmap(); return b; };
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
    this.lightDir2Buf = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.arcCurvatureBuf = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(this.arcCurvatureBuf, 0, new Float32Array([this.material.arcCurvature, 0, 0, 0]));
    this.writeMaterialUBO(); this.writeLightDirUBO(); this.writeLightDir2UBO();

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
    // Separate module for fs_ssao (needs bindings 0-5, separate from composite/debug)
    const ssaoMainMod = d.createShaderModule({ code: ssaoMainWgsl });
    this.ssaoParamsBuf = d.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.screenSizeBuf = d.createBuffer({ size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(this.ssaoParamsBuf, 0, new Float32Array([0.06, 3.0, 0.01, 1.5, 0, 0, 0, 0]));
    this.ssaoProjBuf = d.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    // Cosine-weighted hemisphere kernel: 32 samples.
    // Random points on a disk, projected onto the hemisphere (cosine distribution).
    // More samples near the normal = better quality for the same count.
    // Combined with TBN orientation, this gives a natural AO distribution.
    const KERNEL_SIZE = 48;
    const kernelData = new Float32Array(KERNEL_SIZE * 4);
    for (let i = 0; i < KERNEL_SIZE; i++) {
      // Random point on unit disk (cosine-weighted → sqrt for uniform area)
      const u = Math.random();
      const r = Math.sqrt(u);
      const theta = Math.random() * Math.PI * 2;
      let x = r * Math.cos(theta);
      let y = r * Math.sin(theta);
      let z = Math.sqrt(1.0 - u); // cosine falloff toward horizon
      const len = Math.sqrt(x * x + y * y + z * z);
      x /= len; y /= len; z /= len;
      // Scale by i/KERNEL_SIZE to concentrate samples near center
      const s = i / KERNEL_SIZE;
      const scale = (0.1 + 0.9 * s * s) * Math.random();
      kernelData[i * 4] = x * scale;
      kernelData[i * 4 + 1] = y * scale;
      kernelData[i * 4 + 2] = z * scale;
      kernelData[i * 4 + 3] = 0;
    }
    this.ssaoKernelBuf = d.createBuffer({
      size: kernelData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(this.ssaoKernelBuf.getMappedRange()).set(kernelData);
    this.ssaoKernelBuf.unmap();

    // SSAO bind group layout (group 0): depth + params + screenSize + normal + kernel + proj
    this.ssaoBGL = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });

    // Composite bind group layout (group 1): offscreenColor + occlusion
    this.compositeBGL = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
      ],
    });

    // Contact shadow pipeline: fullscreen quad → contact shadow texture (r32float)
    this.contactShadowMod = d.createShaderModule({ code: contactShadowWgsl });
    this.contactShadowBuf = d.createBuffer({
      size: 192, // invViewProj(64) + viewProj(64) + lightDir(16) + params(16) = 160, padded to 192
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.contactShadowBGL = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });

    // SSAO render pipeline: fullscreen quad → occlusion texture (r32float)
    const ssaoPL = d.createPipelineLayout({ bindGroupLayouts: [this.ssaoBGL] });
    this.ssaoPipe = d.createRenderPipeline({
      layout: ssaoPL,
      vertex: { module: ssaoMainMod, entryPoint: 'vs_fullscreen' },
      fragment: { module: ssaoMainMod, entryPoint: 'fs_ssao', targets: [{ format: 'r32float', writeMask: 15 }] },
      primitive: { topology: 'triangle-list' },
    });

    // ── Shadow mapping ──
    this.shadowMod = d.createShaderModule({ code: shadowWgsl });
    this.shadowVPBuf = d.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.shadowParamsBuf = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(this.shadowParamsBuf, 0, new Float32Array([2.0, 0, 0, 0])); // default softness=2
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
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
      ],
    });
    // Shadow render BGL (used at group 1 of main render pass)
    this.shadowRenderBGL = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });
    // Shadow 2 BGL (group 3 — fill light shadow + direction)
    this.shadowRenderBGL2 = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });
    // Second shadow texture + VP buffer (light 2 — fill from front-right-up)
    this.shadowTex2 = d.createTexture({
      size: [1024, 1024], format: 'depth32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.shadowVPBuf2 = d.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    // ── Render pipelines ──
    const vx = { arrayStride: 24, attributes: [
      { shaderLocation: 0, offset: 0, format: 'float32x3' as const },
      { shaderLocation: 1, offset: 12, format: 'float32x3' as const },
    ]};
    const ds = { format: 'depth32float' as const, depthWriteEnabled: true, depthCompare: 'less' as const };
    const fmt = navigator.gpu.getPreferredCanvasFormat();
    this.offscreenFormat = fmt;
    // IBL bind group layout (group 2) — environment map textures for PBR
    this.iblBGL = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: 'cube' as const } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: 'cube' as const } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d' as const } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      ],
    });
    const rPL = d.createPipelineLayout({ bindGroupLayouts: [this.renderBGL, this.shadowRenderBGL, this.iblBGL, this.shadowRenderBGL2] });

    for (let l = 0; l < 3; l++) this.bodyPipes.push(d.createRenderPipeline({
      layout: rPL, vertex: { module: shaderMod, entryPoint: 'vs_main', buffers: [vx] },
      fragment: { module: shaderMod, entryPoint: 'fs_main', targets: [{ format: fmt, writeMask: 15 }, { format: 'rgba8unorm', writeMask: 15 }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' }, depthStencil: ds,
    }));
    for (let l = 0; l < 2; l++) this.capPipes.push(d.createRenderPipeline({
      layout: rPL, vertex: { module: shaderMod, entryPoint: 'vs_cap', buffers: [vx] },
      fragment: { module: shaderMod, entryPoint: 'fs_main', targets: [{ format: fmt, writeMask: 15 }, { format: 'rgba8unorm', writeMask: 15 }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' }, depthStencil: ds,
    }));
    this.pipeline = this.bodyPipes[0]; this.capPipeline = this.capPipes[0];

    // ── Shadow pipeline (depth-only) ──
    this.shadowTex = d.createTexture({
      size: [1024, 1024], format: 'depth32float',
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
    // Second shadow pipeline (light 2 — fill, same vertex shader, different VP)
    const shadowPL2 = d.createPipelineLayout({ bindGroupLayouts: [this.shadowBGL] });
    this.shadowPipe2 = d.createRenderPipeline({
      layout: shadowPL2,
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
      fragment: { module: this.ssaoMod, entryPoint: 'fs_composite', targets: [{ format: fmt, writeMask: 15 }] },
      primitive: { topology: 'triangle-list' },
    });

    // ── Contact shadow pipeline (fullscreen quad → r32float) ──
    const csPL = d.createPipelineLayout({ bindGroupLayouts: [this.contactShadowBGL] });
    this.contactShadowPipe = d.createRenderPipeline({
      layout: csPL,
      vertex: { module: this.contactShadowMod, entryPoint: 'vs_fullscreen' },
      fragment: { module: this.contactShadowMod, entryPoint: 'fs_contact_shadow', targets: [{ format: 'r32float', writeMask: 15 }] },
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
      fragment: { module: this.ssaoMod, entryPoint: 'fs_debug', targets: [{ format: fmt, writeMask: 15 }] },
      primitive: { topology: 'triangle-list' },
    });
    // Filterable float textures (BRDF LUT, prefilter cubemap faces)
    this.debugFloatBGL = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      ],
    });
    const debugFloatPL = d.createPipelineLayout({ bindGroupLayouts: [this.debugFloatBGL] });
    this.debugFloatPipe = d.createRenderPipeline({
      layout: debugFloatPL,
      vertex: { module: this.ssaoMod, entryPoint: 'vs_fullscreen' },
      fragment: { module: this.ssaoMod, entryPoint: 'fs_debug', targets: [{ format: fmt, writeMask: 15 }] },
      primitive: { topology: 'triangle-list' },
    });
    // Depth textures (SSAO depth, shadow map)
    this.debugDepthBGL = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
      ],
    });
    const debugDepthMod = d.createShaderModule({ code: debugDepthWgsl });
    const debugDepthPL = d.createPipelineLayout({ bindGroupLayouts: [this.debugDepthBGL] });
    this.debugDepthPipe = d.createRenderPipeline({
      layout: debugDepthPL,
      vertex: { module: debugDepthMod, entryPoint: 'vs_fullscreen' },
      fragment: { module: debugDepthMod, entryPoint: 'fs_debug_depth', targets: [{ format: fmt, writeMask: 15 }] },
      primitive: { topology: 'triangle-list' },
    });

    // Color textures (normal, offscreen color, composite) — pass through RGB
    this.debugColorBGL = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      ],
    });
    const debugColorPL = d.createPipelineLayout({ bindGroupLayouts: [this.debugColorBGL] });
    this.debugColorPipe = d.createRenderPipeline({
      layout: debugColorPL,
      vertex: { module: this.ssaoMod, entryPoint: 'vs_fullscreen' },
      fragment: { module: this.ssaoMod, entryPoint: 'fs_copy_color', targets: [{ format: fmt, writeMask: 15 }] },
      primitive: { topology: 'triangle-list' },
    });

    // ── Passthrough copy pipeline (SSAO off: blit offscreen color → swapchain) ──
    // Uses the same single-texture bind group layout as debug (unfilterable-float),
    // but fs_copy_color passes RGBA through instead of grayscale.
    const copyPL = d.createPipelineLayout({ bindGroupLayouts: [this.debugBGL] });
    this.copyPipe = d.createRenderPipeline({
      layout: copyPL,
      vertex: { module: this.ssaoMod, entryPoint: 'vs_fullscreen' },
      fragment: { module: this.ssaoMod, entryPoint: 'fs_copy_color', targets: [{ format: fmt, writeMask: 15 }] },
      primitive: { topology: 'triangle-list' },
    });

    // ── Bilateral blur pipeline ──
    this.blurMod = d.createShaderModule({ code: blurWgsl });
    this.blurBGL = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
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
      fragment: { module: this.blurMod, entryPoint: 'fs_blur', targets: [{ format: 'r32float', writeMask: 15 }] },
      primitive: { topology: 'triangle-list' },
    });

    // ── Velocity pipeline (screen-space motion from depth) ──
    this.velocityMod = d.createShaderModule({ code: velocityWgsl });
    this.velocityBGL = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });
    this.velocityBuf = d.createBuffer({ size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const velocityPL = d.createPipelineLayout({ bindGroupLayouts: [this.velocityBGL] });
    this.velocityPipe = d.createRenderPipeline({
      layout: velocityPL,
      vertex: { module: this.velocityMod, entryPoint: 'vs_fullscreen' },
      fragment: { module: this.velocityMod, entryPoint: 'fs_velocity', targets: [{ format: 'rg16float', writeMask: 15 }] },
      primitive: { topology: 'triangle-list' },
    });

    // ── TAA resolve pipeline ──
    this.taaMod = d.createShaderModule({ code: taaResolveWgsl });
    this.taaBGL = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });
    this.taaParamsBuf = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(this.taaParamsBuf, 0, new Float32Array([0.25, 0, 0, 0]));
    // TAA pipeline uses MRT: color[0] = output, color[1] = history
    const taaPL = d.createPipelineLayout({ bindGroupLayouts: [this.taaBGL] });
    this.taaPipe = d.createRenderPipeline({
      layout: taaPL,
      vertex: { module: this.taaMod, entryPoint: 'vs_fullscreen' },
      fragment: { module: this.taaMod, entryPoint: 'fs_taa', targets: [
        { format: this.offscreenFormat, writeMask: 15 },
        { format: this.taaFormat, writeMask: 15 },
      ]},
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

    // ── Ground plane (4 vertices, correct winding) ──
    const GROUND_SIZE = 250;
    const GROUND_Z = -3;
    const groundVerts = new Float32Array([
      -GROUND_SIZE, -GROUND_SIZE, GROUND_Z,
       GROUND_SIZE, -GROUND_SIZE, GROUND_Z,
       GROUND_SIZE,  GROUND_SIZE, GROUND_Z,
      -GROUND_SIZE,  GROUND_SIZE, GROUND_Z,
    ]);
    const groundIndices = new Uint16Array([0, 1, 2, 0, 2, 3]);
    this._groundIndexCount = 6;
    this.groundVB = d.createBuffer({
      size: groundVerts.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST, mappedAtCreation: true,
    });
    new Float32Array(this.groundVB.getMappedRange()).set(groundVerts);
    this.groundVB.unmap();
    this.groundIB = d.createBuffer({
      size: groundIndices.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST, mappedAtCreation: true,
    });
    new Uint16Array(this.groundIB.getMappedRange()).set(groundIndices);
    this.groundIB.unmap();

    const groundMod = d.createShaderModule({ code: groundWgsl });
    // Ground main pipeline: uses camera at group(0) @binding(0), shadow at group(1)
    this.groundBGL = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });
    const groundPL = d.createPipelineLayout({ bindGroupLayouts: [this.groundBGL, this.shadowRenderBGL, null, this.shadowRenderBGL2] });
    this.groundPipe = d.createRenderPipeline({
      layout: groundPL,
      vertex: { module: groundMod, entryPoint: 'vs_ground', buffers: [{ arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] }] },
      fragment: { module: groundMod, entryPoint: 'fs_ground', targets: [{ format: fmt, writeMask: 15 }, { format: 'rgba8unorm', writeMask: 15 }] },
      primitive: { topology: 'triangle-list', cullMode: 'back' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
    });
    // Ground shadow pipeline: depth-only, uses shadowVP uniform
    this.groundShadowBGL = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
      ],
    });
    const groundShadowPL = d.createPipelineLayout({ bindGroupLayouts: [this.groundShadowBGL] });
    this.groundShadowPipe = d.createRenderPipeline({
      layout: groundShadowPL,
      vertex: { module: groundMod, entryPoint: 'vs_ground_shadow', buffers: [{ arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] }] },
      fragment: { module: groundMod, entryPoint: 'fs_ground_shadow', targets: [] },
      primitive: { topology: 'triangle-list', cullMode: 'back' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
    });

    this.groundBG = d.createBindGroup({
      layout: this.groundBGL,
      entries: [
        { binding: 0, resource: { buffer: this.cameraBuf } },
        { binding: 1, resource: { buffer: this.materialBuf } },
        { binding: 2, resource: { buffer: this.lightDirBuf } },
      ],
    });
    this.groundShadowBG = d.createBindGroup({
      layout: this.groundShadowBGL,
      entries: [{ binding: 0, resource: { buffer: this.shadowVPBuf } }],
    });

    // ── IBL pipeline: load HDR, generate cubemap ──
    this.iblPipeline = new IBLPipeline(d);
    await this.iblPipeline.init('/' + this._envMapUrl);
    this.iblBG = d.createBindGroup({
      layout: this.iblBGL,
      entries: [
        { binding: 0, resource: this.iblPipeline.irradianceMap.createView({ dimension: 'cube' }) },
        { binding: 1, resource: this.iblPipeline.prefilterMap.createView({ dimension: 'cube', mipLevelCount: this.iblPipeline.prefilterMap.mipLevelCount }) },
        { binding: 2, resource: this.iblPipeline.brdfLUT.createView() },
        { binding: 3, resource: this.iblPipeline.iblSampler },
      ],
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
        { binding: 3, resource: { buffer: this.shadowParamsBuf } },
      ],
    });
    // Shadow pass bind group (depth-only, group 0)
    this.shadowPassBG = this.device.createBindGroup({
      layout: this.shadowBGL,
      entries: [
        { binding: 0, resource: { buffer: this.shadowVPBuf } },
        { binding: 1, resource: { buffer: buffers.segmentBuffer } },
        { binding: 2, resource: { buffer: this.arcCurvatureBuf } },
      ],
    });
    // Second shadow (group 3): depth texture + sampler + VP + lightDir2
    this.shadowBG2 = this.device.createBindGroup({
      layout: this.shadowRenderBGL2,
      entries: [
        { binding: 0, resource: this.shadowTex2.createView() },
        { binding: 1, resource: this.shadowSampler },
        { binding: 2, resource: { buffer: this.shadowVPBuf2 } },
        { binding: 3, resource: { buffer: this.lightDir2Buf } },
        { binding: 4, resource: { buffer: this.shadowParamsBuf } },
      ],
    });
    this.shadowPassBG2 = this.device.createBindGroup({
      layout: this.shadowBGL,
      entries: [
        { binding: 0, resource: { buffer: this.shadowVPBuf2 } },
        { binding: 1, resource: { buffer: buffers.segmentBuffer } },
        { binding: 2, resource: { buffer: this.arcCurvatureBuf } },
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
      if (this.iblBG) pass.setBindGroup(2, this.iblBG);
      if (this.shadowBG2) pass.setBindGroup(3, this.shadowBG2);
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
    if (this.iblBG) pass.setBindGroup(2, this.iblBG);
    if (this.shadowBG2) pass.setBindGroup(3, this.shadowBG2);
    pass.setVertexBuffer(0, this.capVB[0]);
    pass.setIndexBuffer(this.capIB[0], 'uint16');
    pass.drawIndexed(this.capIC[0], this.segmentBuffers.capCount);
  }

  writeMaterialUBO() {
    const m = this.material;
    this.device.queue.writeBuffer(this.materialBuf, 0, new Float32Array([
      m.roughness, m.metalness, m.envIntensity, m.specularStrength, m.ambientStrength,
      m.arcCurvature ?? 1, 0, 0,
      m.baseColorTint[0], m.baseColorTint[1], m.baseColorTint[2], m.useRoleColors ?? 1,
    ]));
  }
  writeLightDirUBO() { this.device.queue.writeBuffer(this.lightDirBuf, 0, new Float32Array(this.lightDir)); }
  writeLightDir2UBO() { this.device.queue.writeBuffer(this.lightDir2Buf, 0, new Float32Array(this.lightDir2)); }
  writeArcCurvature() { this.device.queue.writeBuffer(this.arcCurvatureBuf, 0, new Float32Array([this.material.arcCurvature, 0, 0, 0])); }
  writeShadowSoftness() { this.device.queue.writeBuffer(this.shadowParamsBuf, 0, new Float32Array([this.shadowSoftness, 0, 0, 0])); }

  /** Store model bounding box for shadow VP recomputation. */
  setModelBounds(cx: number, cy: number, cz: number, hw: number, hh: number, hd: number) {
    this._modelCenter = [cx, cy, cz];
    this._modelExtent = [hw, hh, hd];
  }

  /** Recompute and upload shadow VP for a given light direction and buffer. */
  writeShadowVP(lx: number, ly: number, lz: number, buf: GPUBuffer) {
    const [cx, cy, cz] = this._modelCenter;
    const [hw, hh, hd] = this._modelExtent;
    const maxDim = Math.max(hw, hh, hd);
    const shadowRadius = maxDim * 5;
    const ll = Math.sqrt(lx*lx + ly*ly + lz*lz);
    const ldx = lx/ll, ldy = ly/ll, ldz = lz/ll;
    const v = new Float64Array(16);
    const p = new Float64Array(16);
    const px = cx - ldx * shadowRadius, py = cy - ldy * shadowRadius, pz = cz - ldz * shadowRadius;
    let fx = cx - px, fy = cy - py, fz = cz - pz;
    const fLen = Math.sqrt(fx*fx + fy*fy + fz*fz);
    if (fLen > 0) { fx /= fLen; fy /= fLen; fz /= fLen; }
    let rx = fy, ry = -fx, rz = 0;
    const rLen = Math.sqrt(rx*rx + ry*ry);
    if (rLen > 0.001) { rx /= rLen; ry /= rLen; } else { rx = 1; ry = 0; }
    const ux = ry*fz - rz*fy, uy = rz*fx - rx*fz, uz = rx*fy - ry*fx;
    v[0]=rx; v[4]=ux; v[8]=-fx; v[12]=-(rx*px+ry*py+rz*pz);
    v[1]=ry; v[5]=uy; v[9]=-fy; v[13]=-(ux*px+uy*py+uz*pz);
    v[2]=rz; v[6]=uz; v[10]=-fz; v[14]=fx*px+fy*py+fz*pz;
    v[3]=0; v[7]=0; v[11]=0; v[15]=1;
    let lmnX=Infinity, lmxX=-Infinity, lmnY=Infinity, lmxY=-Infinity, lmnZ=Infinity, lmxZ=-Infinity;
    for (const sx of [cx-hw, cx+hw]) for (const sy of [cy-hh, cy+hh]) for (const sz of [cz-hd, cz+hd]) {
      const x=v[0]*sx+v[4]*sy+v[8]*sz+v[12], y=v[1]*sx+v[5]*sy+v[9]*sz+v[13], z=v[2]*sx+v[6]*sy+v[10]*sz+v[14];
      if(x<lmnX)lmnX=x;if(x>lmxX)lmxX=x;if(y<lmnY)lmnY=y;if(y>lmxY)lmxY=y;if(z<lmnZ)lmnZ=z;if(z>lmxZ)lmxZ=z;
    }
    if(0<lmnZ)lmnZ=0;if(0>lmxZ)lmxZ=0;
    const pad=4.0, hlw=Math.max(Math.abs(lmnX),Math.abs(lmxX))*pad, hlh=Math.max(Math.abs(lmnY),Math.abs(lmxY))*pad;
    const zn=lmnZ-0.5, zf=lmxZ+0.5, dr=zf-zn;
    p[0]=1/hlw;p[4]=0;p[8]=0;p[12]=0;p[1]=0;p[5]=1/hlh;p[9]=0;p[13]=0;
    p[2]=0;p[6]=0;p[10]=1/dr;p[14]=-zn/dr;p[3]=0;p[7]=0;p[11]=0;p[15]=1;
    const svp = new Float32Array(16);
    for(let col=0;col<4;col++) for(let row=0;row<4;row++)
      svp[col*4+row]=p[row]*v[col*4]+p[4+row]*v[col*4+1]+p[8+row]*v[col*4+2]+p[12+row]*v[col*4+3];
    this.device.queue.writeBuffer(buf, 0, svp);
  }
  /** Write SSAO params buffer (radius, intensity, bias, power). Leaves camera params at offset 16 intact. */
  writeSSAOParams() {
    const d = new Float32Array([this.ssaoRadius, this.ssaoIntensity, 0.01, 1.5]);
    this.device.queue.writeBuffer(this.ssaoParamsBuf, 0, d);
  }

  /** Reload the IBL environment map from a new HDRI. Re-generates cubemap, irradiance, prefilter, BRDF LUT. */
  async setEnvMap(url: string) {
    this._envMapUrl = url;
    if (!this.iblPipeline) return;
    this.iblPipeline.dispose();
    this.iblPipeline = new IBLPipeline(this.device);
    await this.iblPipeline.init('/' + url);
    this.iblBG = this.device.createBindGroup({
      layout: this.iblBGL,
      entries: [
        { binding: 0, resource: this.iblPipeline.irradianceMap.createView({ dimension: 'cube' }) },
        { binding: 1, resource: this.iblPipeline.prefilterMap.createView({ dimension: 'cube', mipLevelCount: this.iblPipeline.prefilterMap.mipLevelCount }) },
        { binding: 2, resource: this.iblPipeline.brdfLUT.createView() },
        { binding: 3, resource: this.iblPipeline.iblSampler },
      ],
    });
  }
  writeCameraUBO(camera: OrbitCamera) {
    const d = new Float32Array(36);
    d.set(camera.viewProj, 0);
    d.set(camera.viewMat, 16);
    d[32] = camera.position[0]; d[33] = camera.position[1]; d[34] = camera.position[2];
    this.device.queue.writeBuffer(this.cameraBuf, 0, d);
    // Write velocity UBO (invViewProj + prevViewProj) for TAA
    this.writeVelocity(camera);
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
    for (const t of [this.offscreenColorTex, this.normalTex, this.ssaoDepthTex, this.ssaoOcclusionTex, this.blurTempTex, this.contactShadowTex]) t?.destroy();

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
    // Contact shadow texture (r32float: 1.0 = lit, 0.0 = fully shadowed)
    this.contactShadowTex = d.createTexture({
      size: [w, h], format: 'r32float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });

    // Write screen size uniform
    d.queue.writeBuffer(this.screenSizeBuf, 0, new Float32Array([w, h]));

    // SSAO bind group (group 0): depth texture + params + screenSize + normal + kernel + proj
    this.ssaoBG = d.createBindGroup({
      layout: this.ssaoBGL,
      entries: [
        { binding: 0, resource: this.ssaoDepthTex.createView() },
        { binding: 1, resource: { buffer: this.ssaoParamsBuf } },
        { binding: 2, resource: { buffer: this.screenSizeBuf } },
        { binding: 3, resource: this.normalTex.createView() },
        { binding: 4, resource: { buffer: this.ssaoKernelBuf } },
        { binding: 5, resource: { buffer: this.ssaoProjBuf } },
      ],
    });

    // Composite bind group (group 1): offscreen color + occlusion + contact shadow
    this.compositeBG = d.createBindGroup({
      layout: this.compositeBGL,
      entries: [
        { binding: 0, resource: this.offscreenColorTex.createView() },
        { binding: 1, resource: this.ssaoOcclusionTex.createView() },
        { binding: 2, resource: this.contactShadowTex.createView() },
      ],
    });

    // Contact shadow bind group: depth + params
    this.contactShadowBG = d.createBindGroup({
      layout: this.contactShadowBGL,
      entries: [
        { binding: 0, resource: this.ssaoDepthTex.createView() },
        { binding: 1, resource: { buffer: this.contactShadowBuf } },
      ],
    });

    this.ssaoWidth = w;
    this.ssaoHeight = h;

    // Also resize TAA textures
    this.resizeTAA(w, h);

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
        timestampWrites: this._gpuQuerySet ? { querySet: this._gpuQuerySet, beginningOfPassWriteIndex: this._gpuQueryIdx++, endOfPassWriteIndex: this._gpuQueryIdx++ } : undefined,
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
        timestampWrites: this._gpuQuerySet ? { querySet: this._gpuQuerySet, beginningOfPassWriteIndex: this._gpuQueryIdx++, endOfPassWriteIndex: this._gpuQueryIdx++ } : undefined,
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
  }

  /** Create/recreate TAA textures at given resolution. */
  resizeTAA(w: number, h: number) {
    const d = this.device;
    for (const t of [this.velocityTex, this.compositeTex, ...this.historyTex]) t?.destroy();
    this.velocityTex = d.createTexture({
      size: [w, h], format: 'rg16float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.compositeTex = d.createTexture({
      size: [w, h], format: this.offscreenFormat, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    // Two history textures for ping-pong (read one, write the other) — float precision
    this.historyTex = [0, 1].map(() => d.createTexture({
      size: [w, h], format: this.taaFormat, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    }));
    this._historyIndex = 0;

    // Velocity bind group: depth + params
    this.velocityBG = d.createBindGroup({
      layout: this.velocityBGL,
      entries: [
        { binding: 0, resource: this.ssaoDepthTex.createView() },
        { binding: 1, resource: { buffer: this.velocityBuf } },
      ],
    });

    // TAA bind groups: one per history texture (read from history[i], write to history[1-i])
    this.taaBG = [0, 1].map(i => d.createBindGroup({
      layout: this.taaBGL,
      entries: [
        { binding: 0, resource: this.compositeTex.createView() },
        { binding: 1, resource: this.velocityTex.createView() },
        { binding: 2, resource: this.historyTex[i].createView() },
        { binding: 3, resource: { buffer: this.taaParamsBuf } },
      ],
    }));
  }

  /** Write velocity UBO (invViewProj + prevViewProj). */
  writeVelocity(camera: import('./camera').OrbitCamera) {
    const d = new Float32Array(32);
    d.set(camera.invViewProj, 0);
    d.set(camera.prevViewProj, 16);
    this.device.queue.writeBuffer(this.velocityBuf, 0, d);
  }

  dispatchVelocity(enc: GPUCommandEncoder) {
    if (!this.velocityBG) return;
    const h = this._gpuQuerySet ? { querySet: this._gpuQuerySet, beginningOfPassWriteIndex: this._gpuQueryIdx++, endOfPassWriteIndex: this._gpuQueryIdx++ } : undefined;
    const pass = enc.beginRenderPass({
      timestampWrites: h,
      colorAttachments: [{ view: this.velocityTex.createView(), loadOp: 'clear', storeOp: 'store' }],
    });
    pass.setPipeline(this.velocityPipe);
    pass.setBindGroup(0, this.velocityBG);
    pass.draw(3);
    pass.end();
  }

  /** TAA resolve: blend composite color + velocity + history → swapchain + new history. */
  dispatchTAA(enc: GPUCommandEncoder, outView: GPUTextureView) {
    if (!this.taaBG.length) return;
    const readIdx = this._historyIndex;
    const writeIdx = 1 - readIdx;
    const h = this._gpuQuerySet ? { querySet: this._gpuQuerySet, beginningOfPassWriteIndex: this._gpuQueryIdx++, endOfPassWriteIndex: this._gpuQueryIdx++ } : undefined;
    const pass = enc.beginRenderPass({
      timestampWrites: h,
      colorAttachments: [
        { view: outView, loadOp: 'clear', storeOp: 'store' },
        { view: this.historyTex[writeIdx].createView(), loadOp: 'clear', storeOp: 'store' },
      ],
    });
    pass.setPipeline(this.taaPipe);
    pass.setBindGroup(0, this.taaBG[readIdx]); // reads from history[readIdx]
    pass.draw(3);
    pass.end();
    this._historyIndex = writeIdx; // next frame reads from what we just wrote
    this.taaFrame++;
  }

  /** Halton(2, i) for X jitter. */
  static halton2(i: number): number {
    let f = 1, r = 0;
    while (i > 0) { f /= 2; r += (i & 1) * f; i >>= 1; }
    return r - 0.5;
  }
  static halton3(i: number): number {
    let f = 1, r = 0;
    while (i > 0) { f /= 3; r += (i % 3) * f; i = Math.floor(i / 3); }
    return r - 0.5;
  }
  /** 8-sample periodic jitter pattern (first 8 Halton(2,3) samples, shifted to [-0.5, 0.5]).
   *  Cycling ensures the TAA converges to the exact same set of sub-pixel positions each
   *  cycle, eliminating beat-frequency oscillation on high-frequency detail (moire, layer lines). */
  static taaJitterPattern: [number, number][] = Array.from({ length: 8 }, (_, i) =>
    [SlicedPipeline.halton2(i), SlicedPipeline.halton3(i)]
  );
  /** Get the jitter offset for a given frame index (wraps through the pattern). */
  static getTAAJitter(frame: number): [number, number] {
    return SlicedPipeline.taaJitterPattern[frame % SlicedPipeline.taaJitterPattern.length];
  }
  /** Apply sub-pixel jitter to a column-major projection matrix in-place. */
  static jitterProj(proj: Float32Array, jitterX: number, jitterY: number, w: number, h: number) {
    proj[8] += (jitterX * 2) / w;
    proj[9] += (jitterY * 2) / h;
  }

  /** Render shadow map (depth-only from light POV). */
  renderShadowMap(enc: GPUCommandEncoder) {
    if (!this.segmentBuffers) return;
    enc.pushDebugGroup('shadow-map');
    const h = this._gpuQuerySet ? { querySet: this._gpuQuerySet, beginningOfPassWriteIndex: this._gpuQueryIdx++, endOfPassWriteIndex: this._gpuQueryIdx++ } : undefined;
    const pass = enc.beginRenderPass({
      timestampWrites: h,
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
    // Draw LOD 1 body geometry (8-vertex cross-section) as shadow casters.
    // LOD 1 is sufficient for shadow depth — simpler than LOD 0 but fills
    // the same bounding area, and we don't need cap silhouettes in the shadow.
    pass.setVertexBuffer(0, this.bodyVB[1]);
    pass.setIndexBuffer(this.bodyIB[1], 'uint16');
    pass.drawIndexed(this.bodyIC[1], this.segmentBuffers.count);
    pass.end();
    enc.popDebugGroup();
  }

  /** Render shadow map for light 2 (fill light). */
  renderShadowMap2(enc: GPUCommandEncoder) {
    if (!this.segmentBuffers) return;
    enc.pushDebugGroup('shadow-map2');
    const h = this._gpuQuerySet ? { querySet: this._gpuQuerySet, beginningOfPassWriteIndex: this._gpuQueryIdx++, endOfPassWriteIndex: this._gpuQueryIdx++ } : undefined;
    const pass = enc.beginRenderPass({
      timestampWrites: h,
      colorAttachments: [],
      depthStencilAttachment: {
        view: this.shadowTex2.createView(),
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    });
    pass.setPipeline(this.shadowPipe2);
    pass.setBindGroup(0, this.shadowPassBG2);
    pass.setVertexBuffer(0, this.bodyVB[1]);
    pass.setIndexBuffer(this.bodyIB[1], 'uint16');
    pass.drawIndexed(this.bodyIC[1], this.segmentBuffers.count);
    pass.end();
    enc.popDebugGroup();
  }

  /** Render screen-space contact shadows (ray-march against depth buffer). */
  renderContactShadow(enc: GPUCommandEncoder) {
    if (!this.contactShadowBG) return;
    const h = this._gpuQuerySet ? { querySet: this._gpuQuerySet, beginningOfPassWriteIndex: this._gpuQueryIdx++, endOfPassWriteIndex: this._gpuQueryIdx++ } : undefined;
    const pass = enc.beginRenderPass({
      timestampWrites: h,
      colorAttachments: [{
        view: this.contactShadowTex.createView(),
        loadOp: 'clear',
        clearValue: { r: 1.0, g: 0, b: 0, a: 0 },
        storeOp: 'store',
      }],
    });
    pass.setPipeline(this.contactShadowPipe);
    pass.setBindGroup(0, this.contactShadowBG);
    pass.draw(3);
    pass.end();
  }

  /** Write contact shadow uniform buffer (invViewProj + viewProj + lightDir + params). */
  writeContactShadow(viewProj: Float32Array, invViewProj: Float32Array, lightDir: [number,number,number,number]) {
    const d = new Float32Array(48); // 16+16+4+4 = 40 floats, padded
    d.set(invViewProj, 0);
    d.set(viewProj, 16);
    d[32] = lightDir[0]; d[33] = lightDir[1]; d[34] = lightDir[2]; d[35] = this.contactShadowStrength;
    d[36] = this.contactShadowDist;   // maxDist
    d[37] = 32;     // stepCount
    d[38] = 0.01;   // thickness — depth rejection threshold
    d[39] = 0.1;    // edgeFadeDist — fade to no shadow in outer 10% of screen
    d[40] = this.contactShadowStrength; // stored at index 40 (after 16+16+4+4 = 40 floats)
    this.device.queue.writeBuffer(this.contactShadowBuf, 0, d);
  }

  /** Draw ground plane in the main offscreen pass. */
  drawGround(pass: GPURenderPassEncoder) {
    if (!this.shadowBG) return;
    pass.setPipeline(this.groundPipe);
    pass.setBindGroup(0, this.groundBG);
    pass.setBindGroup(1, this.shadowBG);
    if (this.shadowBG2) pass.setBindGroup(3, this.shadowBG2);
    pass.setVertexBuffer(0, this.groundVB);
    pass.setIndexBuffer(this.groundIB, 'uint16');
    pass.drawIndexed(this._groundIndexCount, 1);
  }

  _groundZ = -3;
  _groundIndexCount = 0;

  /** Move the ground plane to a new Z position. Rebuilds the vertex buffer. */
  setGroundZ(z: number) {
    this._groundZ = z;
    const verts = new Float32Array([-250, -250, z, 250, -250, z, 250, 250, z, -250, 250, z]);
    this.device.queue.writeBuffer(this.groundVB, 0, verts);
  }

  /** Render a debug preview of one internal texture to a render pass. */
  renderDebugView(pass: GPURenderPassEncoder, mode: string) {
    let view: GPUTextureView | undefined;
    let depthMode = false;
    let floatMode = false;
    let colorMode = false;
    switch (mode) {
      case 'depth':
        view = this.ssaoDepthTex.createView();
        depthMode = true;
        break;
      case 'occlusion':
        view = this.ssaoOcclusionTex.createView();
        break;
      case 'blur-temp':
        view = this.blurTempTex?.createView();
        if (!view) return;
        break;
      case 'brdf-lut':
        view = this.iblPipeline?.brdfLUT.createView();
        if (!view) return;
        floatMode = true;
        break;
      case 'prefilter-up':
        view = this.iblPipeline?.prefilterMap.createView({ dimension: '2d', baseArrayLayer: 4, arrayLayerCount: 1, baseMipLevel: 0, mipLevelCount: 1 });
        if (!view) return;
        floatMode = true;
        break;
      case 'prefilter-fwd':
        view = this.iblPipeline?.prefilterMap.createView({ dimension: '2d', baseArrayLayer: 2, arrayLayerCount: 1, baseMipLevel: 0, mipLevelCount: 1 });
        if (!view) return;
        floatMode = true;
        break;
      case 'prefilter-down':
        view = this.iblPipeline?.prefilterMap.createView({ dimension: '2d', baseArrayLayer: 5, arrayLayerCount: 1, baseMipLevel: 0, mipLevelCount: 1 });
        if (!view) return;
        floatMode = true;
        break;
      case 'source-up':
        view = this.iblPipeline?.envCubemap.createView({ dimension: '2d', baseArrayLayer: 4, arrayLayerCount: 1, baseMipLevel: 0, mipLevelCount: 1 });
        if (!view) return;
        floatMode = true;
        break;
      case 'source-fwd':
        view = this.iblPipeline?.envCubemap.createView({ dimension: '2d', baseArrayLayer: 2, arrayLayerCount: 1, baseMipLevel: 0, mipLevelCount: 1 });
        if (!view) return;
        floatMode = true;
        break;
      case 'source-down':
        view = this.iblPipeline?.envCubemap.createView({ dimension: '2d', baseArrayLayer: 5, arrayLayerCount: 1, baseMipLevel: 0, mipLevelCount: 1 });
        if (!view) return;
        floatMode = true;
        break;
      case 'color':
        view = this.offscreenColorTex.createView();
        colorMode = true;
        break;
      case 'normal':
        view = this.normalTex.createView();
        colorMode = true;
        break;
      case 'velocity':
        view = this.velocityTex?.createView();
        if (!view) return;
        break;
      case 'composite-taa':
        view = this.compositeTex?.createView();
        if (!view) return;
        colorMode = true;
        break;
      case 'shadow':
        view = this.shadowTex.createView();
        depthMode = true;
        break;
      case 'shadow2':
        view = this.shadowTex2.createView();
        depthMode = true;
        break;
      default:
        return;
    }
    const bg = this.device.createBindGroup({
      layout: colorMode ? this.debugColorBGL : (floatMode ? this.debugFloatBGL : (depthMode ? this.debugDepthBGL : this.debugBGL)),
      entries: [{ binding: 0, resource: view! }],
    });
    pass.setPipeline(colorMode ? this.debugColorPipe : (floatMode ? this.debugFloatPipe : (depthMode ? this.debugDepthPipe : this.debugPipe)));
    pass.setBindGroup(0, bg);
    pass.draw(3);
  }
  composite(pass: GPURenderPassEncoder) {
    if (!this.compositeBG) return;
    pass.setPipeline(this.compositePipe);
    pass.setBindGroup(1, this.compositeBG);
    pass.draw(3); // fullscreen triangle
  }

  /** Copy offscreen color texture to a swapchain pass (SSAO-disabled path). */
  copyToSwapchain(pass: GPURenderPassEncoder) {
    const bg = this.device.createBindGroup({
      layout: this.debugBGL,
      entries: [{ binding: 0, resource: this.offscreenColorTex.createView() }],
    });
    pass.setPipeline(this.copyPipe);
    pass.setBindGroup(0, bg);
    pass.draw(3);
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
    for (const t of [this.offscreenColorTex, this.normalTex, this.ssaoDepthTex, this.ssaoOcclusionTex, this.blurTempTex, this.velocityTex, this.compositeTex, ...this.historyTex, this.contactShadowTex]) t?.destroy();
    this.ssaoParamsBuf?.destroy();
    this.ssaoKernelBuf?.destroy();
    this.ssaoProjBuf?.destroy();
    this.shadowTex?.destroy();
    this.shadowVPBuf?.destroy();
    this.shadowTex2?.destroy();
    this.shadowVPBuf2?.destroy();
    this.contactShadowBuf?.destroy();
    this.lightDir2Buf?.destroy();
    this.arcCurvatureBuf?.destroy();
  }
}
