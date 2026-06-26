/**
 * IBL pipeline — loads HDR, generates cubemap + irradiance + prefilter + BRDF LUT.
 *
 * Usage:
 *   const ibl = new IBLPipeline(device);
 *   await ibl.init('horn-koppe_spring_1k.hdr');
 *   // Then bind ibl.irradianceMap, ibl.prefilterMap, ibl.brdfLUT in your PBR shader.
 */

import { loadHDR } from './hdr';
import equirectToCubemapWgsl from './shaders/equirect_to_cubemap.wgsl?raw';
import irradianceConvolutionWgsl from './shaders/irradiance_convolution.wgsl?raw';
import iblSharedWgsl from './shaders/ibl_shared.wgsl?raw';
import prefilterEnvmapWgsl from './shaders/prefilter_envmap.wgsl?raw';
import brdfLutWgsl from './shaders/brdf_lut.wgsl?raw';

const prefilterWgsl = iblSharedWgsl + prefilterEnvmapWgsl;
const brdfWgsl = iblSharedWgsl + brdfLutWgsl;

const CUBE_SIZE = 512;

// Unit cube mesh — 36 vertices (6 faces × 2 tris × 3 verts), vec4 position.
// prettier-ignore
const CUBE_VERTS = new Float32Array([
   1, -1,  1, 1,   -1, -1,  1, 1,   -1, -1, -1, 1,
   1, -1, -1, 1,    1, -1,  1, 1,   -1, -1, -1, 1,
   1,  1,  1, 1,    1, -1,  1, 1,    1, -1, -1, 1,
   1,  1, -1, 1,    1,  1,  1, 1,    1, -1, -1, 1,
  -1,  1,  1, 1,    1,  1,  1, 1,    1,  1, -1, 1,
  -1,  1, -1, 1,   -1,  1,  1, 1,    1,  1, -1, 1,
  -1, -1,  1, 1,   -1,  1,  1, 1,   -1,  1, -1, 1,
  -1, -1, -1, 1,   -1, -1,  1, 1,   -1,  1, -1, 1,
   1,  1,  1, 1,   -1,  1,  1, 1,   -1, -1,  1, 1,
  -1, -1,  1, 1,    1, -1,  1, 1,    1,  1,  1, 1,
   1, -1, -1, 1,   -1, -1, -1, 1,   -1,  1, -1, 1,
   1,  1, -1, 1,    1, -1, -1, 1,   -1,  1, -1, 1,
]);

// 6 face view matrices following the cubemap convention:
// camera at origin looking along each axis direction.
// These are pure rotation matrices (no translation since eye = origin).
function makeFaceMatrices(): Float32Array[] {
  const proj = perspective90(1);
  // Direct cubemap face view matrices (column-major), camera at origin.
  // For each face: fwd = direction, up = face-up, right = cross(up, fwd).
  // view_z = right.z*x + up.z*y + (-fwd.z)*z must equal -dot(vertex, fwd)
  // prettier-ignore
  const views = [
    // +X: right        right=(0,0,-1)  up=(0,1,0)  -fwd=(-1,0,0)
    new Float32Array([ 0, 0,-1, 0,   0, 1, 0, 0,  -1, 0, 0, 0,   0, 0, 0, 1 ]),
    // -X: left         right=(0,0,1)   up=(0,1,0)  -fwd=(1,0,0)
    new Float32Array([ 0, 0, 1, 0,   0, 1, 0, 0,   1, 0, 0, 0,   0, 0, 0, 1 ]),
    // +Y: up           right=(1,0,0)  up=(0,0,-1)  -fwd=(0,-1,0)
    new Float32Array([ 1, 0, 0, 0,   0, 0,-1, 0,   0,-1, 0, 0,   0, 0, 0, 1 ]),
    // -Y: down         right=(1,0,0)  up=(0,0,1)   -fwd=(0,1,0)
    new Float32Array([ 1, 0, 0, 0,   0, 0, 1, 0,   0, 1, 0, 0,   0, 0, 0, 1 ]),
    // +Z: forward      right=(1,0,0)  up=(0,1,0)   -fwd=(0,0,-1)
    new Float32Array([ 1, 0, 0, 0,   0, 1, 0, 0,   0, 0,-1, 0,   0, 0, 0, 1 ]),
    // -Z: backward     right=(-1,0,0) up=(0,1,0)   -fwd=(0,0,1)
    new Float32Array([-1, 0, 0, 0,   0, 1, 0, 0,   0, 0, 1, 0,   0, 0, 0, 1 ]),
  ];
  return views.map(v => mul44(proj, v));
}

/** 90° perspective projection, column-major Float32Array. */
function perspective90(aspect: number): Float32Array {
  const f = 1 / Math.tan(Math.PI / 4); // tan(45°) = 1
  const nf = 1 / (0.1 - 10);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, 10 * nf, -1,
    0, 0, 0.1 * 10 * nf, 0,
  ]);
}

/** Column-major 4×4 multiply: A × B. */
function mul44(a: Float32Array, b: Float32Array): Float32Array {
  const r = new Float32Array(16);
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      r[col * 4 + row] =
        a[row] * b[col * 4] + a[4 + row] * b[col * 4 + 1] +
        a[8 + row] * b[col * 4 + 2] + a[12 + row] * b[col * 4 + 3];
    }
  }
  return r;
}

export class IBLPipeline {
  device: GPUDevice;

  // Generated textures (set after init)
  envCubemap!: GPUTexture;
  irradianceMap!: GPUTexture;
  prefilterMap!: GPUTexture;
  brdfLUT!: GPUTexture;
  iblSampler!: GPUSampler;

  private cubeVB!: GPUBuffer;
  private faceMatrices!: Float32Array[];

  constructor(device: GPUDevice) {
    this.device = device;
  }

  async init(hdrUrl: string) {
    const d = this.device;

    // Create placeholder textures for the full IBL pipeline.
    // Real generation will replace these in later steps.
    // Must be 6-layer 2D arrays so their views can use dimension 'cube'.
    const stubSize = 1;
    this.irradianceMap = d.createTexture({
      dimension: '2d',
      size: { width: stubSize, height: stubSize, depthOrArrayLayers: 6 },
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this.prefilterMap = d.createTexture({
      dimension: '2d',
      size: { width: stubSize, height: stubSize, depthOrArrayLayers: 6 },
      format: 'rgba8unorm',
      mipLevelCount: 1,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this.brdfLUT = d.createTexture({
      size: { width: stubSize, height: stubSize },
      format: 'rg16float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });

    // Cube vertex buffer
    this.cubeVB = d.createBuffer({
      size: CUBE_VERTS.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(this.cubeVB.getMappedRange()).set(CUBE_VERTS);
    this.cubeVB.unmap();

    this.faceMatrices = makeFaceMatrices();

    // Create sampler used for all IBL textures
    this.iblSampler = d.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      mipmapFilter: 'linear',
    });

    // Step 1: Load HDR and create equirect texture
    const hdr = await loadHDR(hdrUrl);
    const equirectTex = d.createTexture({
      size: [hdr.width, hdr.height],
      format: 'rgba16float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    d.queue.writeTexture(
      { texture: equirectTex },
      hdr.data.buffer,
      { bytesPerRow: hdr.width * 8, rowsPerImage: hdr.height },
      [hdr.width, hdr.height],
    );

    // Step 2: Convert equirect → cubemap
    this.envCubemap = this.renderEquirectToCubemap(equirectTex);
    equirectTex.destroy();

    // Step 3: Diffuse irradiance convolution
    this.irradianceMap.destroy();
    this.irradianceMap = this.computeIrradiance(this.envCubemap);

    // Step 4: Specular prefilter (GGX importance sampling, 5 mip levels)
    this.prefilterMap.destroy();
    this.prefilterMap = this.computePrefilter(this.envCubemap);

    // Step 5: BRDF LUT (environment-independent, pre-computed once)
    this.brdfLUT.destroy();
    this.brdfLUT = this.computeBRDFLUT();
  }

  /** Render equirectangular HDR into a 6-face rgba16float cubemap (512×512). */
  private renderEquirectToCubemap(equirectTex: GPUTexture): GPUTexture {
    const d = this.device;
    const size = CUBE_SIZE;

    const cubemap = d.createTexture({
      dimension: '2d',
      size: { width: size, height: size, depthOrArrayLayers: 6 },
      format: 'rgba16float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
    });

    const mod = d.createShaderModule({ code: equirectToCubemapWgsl });
    const bgl = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      ],
    });
    const pipe = d.createRenderPipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [bgl] }),
      vertex: {
        module: mod, entryPoint: 'vs_main',
        buffers: [{ arrayStride: 16, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x4' }] }],
      },
      fragment: {
        module: mod, entryPoint: 'fs_main',
        targets: [{ format: 'rgba16float' }],
      },
      primitive: { topology: 'triangle-list' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    });

    const depthTex = d.createTexture({
      size: [size, size], format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    const depthView = depthTex.createView();

    const uniformBuf = d.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    for (let face = 0; face < 6; face++) {
      d.queue.writeBuffer(uniformBuf, 0, this.faceMatrices[face]);

      const bg = d.createBindGroup({
        layout: bgl,
        entries: [
          { binding: 0, resource: { buffer: uniformBuf } },
          { binding: 1, resource: equirectTex.createView() },
          { binding: 2, resource: d.createSampler({ magFilter: 'linear', minFilter: 'linear' }) },
        ],
      });

      const enc = d.createCommandEncoder();
      const pass = enc.beginRenderPass({
        colorAttachments: [{
          view: cubemap.createView({ dimension: '2d', baseArrayLayer: face, arrayLayerCount: 1 }),
          loadOp: 'clear', storeOp: 'store',
        }],
        depthStencilAttachment: { view: depthView, depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
      });
      pass.setPipeline(pipe);
      pass.setViewport(0, 0, size, size, 0, 1);
      pass.setVertexBuffer(0, this.cubeVB);
      pass.setBindGroup(0, bg);
      pass.draw(36);
      pass.end();
      d.queue.submit([enc.finish()]);
    }

    depthTex.destroy();
    return cubemap;
  }

  /** Convolve env cubemap into a 32×32 diffuse irradiance cubemap via hemisphere integration. */
  private computeIrradiance(envCubemap: GPUTexture): GPUTexture {
    const d = this.device;
    const size = 32;

    const tex = d.createTexture({
      dimension: '2d',
      size: { width: size, height: size, depthOrArrayLayers: 6 },
      format: 'rgba16float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
    });

    const mod = d.createShaderModule({ code: irradianceConvolutionWgsl });
    const bgl = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { viewDimension: 'cube' as const } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      ],
    });
    const pipe = d.createRenderPipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [bgl] }),
      vertex: {
        module: mod, entryPoint: 'vs_main',
        buffers: [{ arrayStride: 16, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x4' }] }],
      },
      fragment: {
        module: mod, entryPoint: 'fs_main',
        targets: [{ format: 'rgba16float' }],
      },
      primitive: { topology: 'triangle-list' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    });

    const depthTex = d.createTexture({
      size: [size, size], format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    const depthView = depthTex.createView();
    const uniformBuf = d.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const envView = envCubemap.createView({ dimension: 'cube' });

    for (let face = 0; face < 6; face++) {
      d.queue.writeBuffer(uniformBuf, 0, this.faceMatrices[face]);

      const bg = d.createBindGroup({
        layout: bgl,
        entries: [
          { binding: 0, resource: { buffer: uniformBuf } },
          { binding: 1, resource: envView },
          { binding: 2, resource: this.iblSampler },
        ],
      });

      const enc = d.createCommandEncoder();
      const pass = enc.beginRenderPass({
        colorAttachments: [{
          view: tex.createView({ dimension: '2d', baseArrayLayer: face, arrayLayerCount: 1 }),
          loadOp: 'clear', storeOp: 'store',
        }],
        depthStencilAttachment: { view: depthView, depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
      });
      pass.setPipeline(pipe);
      pass.setViewport(0, 0, size, size, 0, 1);
      pass.setVertexBuffer(0, this.cubeVB);
      pass.setBindGroup(0, bg);
      pass.draw(36);
      pass.end();
      d.queue.submit([enc.finish()]);
    }

    depthTex.destroy();
    return tex;
  }

  /** Prefilter env cubemap into a mipmapped specular cubemap via GGX importance sampling. */
  private computePrefilter(envCubemap: GPUTexture): GPUTexture {
    const d = this.device;
    const size = 256;
    const levels = 5;

    const tex = d.createTexture({
      dimension: '2d',
      size: { width: size, height: size, depthOrArrayLayers: 6 },
      format: 'rgba16float',
      mipLevelCount: levels,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
    });

    const mod = d.createShaderModule({ code: prefilterWgsl });
    const bgl = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { viewDimension: 'cube' as const } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      ],
    });
    const pipe = d.createRenderPipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [bgl] }),
      vertex: {
        module: mod, entryPoint: 'vs_main',
        buffers: [{ arrayStride: 16, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x4' }] }],
      },
      fragment: {
        module: mod, entryPoint: 'fs_main',
        targets: [{ format: 'rgba16float' }],
      },
      primitive: { topology: 'triangle-list' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    });

    const envView = envCubemap.createView({ dimension: 'cube' });
    const uniformBuf = d.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const roughnessBuf = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const depthTex = d.createTexture({
      size: [size, size], format: 'depth24plus',
      mipLevelCount: levels,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });

    for (let mip = 0; mip < levels; mip++) {
      const mipSize = Math.max(1, size >> mip);
      const roughnessVal = mip / (levels - 1);
      d.queue.writeBuffer(roughnessBuf, 0, new Float32Array([roughnessVal, 0, 0, 0]));

      const depthView = depthTex.createView({ baseMipLevel: mip, mipLevelCount: 1 });

      for (let face = 0; face < 6; face++) {
        d.queue.writeBuffer(uniformBuf, 0, this.faceMatrices[face]);

        const bg = d.createBindGroup({
          layout: bgl,
          entries: [
            { binding: 0, resource: { buffer: uniformBuf } },
            { binding: 1, resource: { buffer: roughnessBuf } },
            { binding: 2, resource: envView },
            { binding: 3, resource: this.iblSampler },
          ],
        });

        const enc = d.createCommandEncoder();
        const pass = enc.beginRenderPass({
          colorAttachments: [{
            view: tex.createView({ dimension: '2d', baseArrayLayer: face, arrayLayerCount: 1, baseMipLevel: mip, mipLevelCount: 1 }),
            loadOp: 'clear', storeOp: 'store',
          }],
          depthStencilAttachment: { view: depthView, depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
        });
        pass.setPipeline(pipe);
        pass.setViewport(0, 0, mipSize, mipSize, 0, 1);
        pass.setScissorRect(0, 0, mipSize, mipSize);
        pass.setVertexBuffer(0, this.cubeVB);
        pass.setBindGroup(0, bg);
        pass.draw(36);
        pass.end();
        d.queue.submit([enc.finish()]);
      }
    }

    depthTex.destroy();
    return tex;
  }

  /** Compute the BRDF integration LUT (512×512 rg16float, environment-independent). */
  private computeBRDFLUT(): GPUTexture {
    const d = this.device;
    const size = 512;

    const tex = d.createTexture({
      size: { width: size, height: size },
      format: 'rg16float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
    });

    const mod = d.createShaderModule({ code: brdfWgsl });
    const bgl = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });
    const pipe = d.createRenderPipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [bgl] }),
      vertex: {
        module: mod, entryPoint: 'vs_main',
        buffers: [{
          arrayStride: 20, // 3 floats pos + 2 floats uv
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x3' },
            { shaderLocation: 1, offset: 12, format: 'float32x2' },
          ],
        }],
      },
      fragment: {
        module: mod, entryPoint: 'fs_main',
        targets: [{ format: 'rg16float' }],
      },
      primitive: { topology: 'triangle-list' },
    });

    // Fullscreen quad vertices (pos.xyz + uv.uv)
    // prettier-ignore
    const quadVerts = new Float32Array([
      -1, -1, 0,  0, 0,
       1, -1, 0,  1, 0,
       1,  1, 0,  1, 1,
      -1, -1, 0,  0, 0,
       1,  1, 0,  1, 1,
      -1,  1, 0,  0, 1,
    ]);
    const vb = d.createBuffer({
      size: quadVerts.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(vb.getMappedRange()).set(quadVerts);
    vb.unmap();

    const uniformBuf = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const bg = d.createBindGroup({
      layout: bgl,
      entries: [{ binding: 0, resource: { buffer: uniformBuf } }],
    });

    const enc = d.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [{ view: tex.createView(), loadOp: 'clear', storeOp: 'store' }],
    });
    pass.setPipeline(pipe);
    pass.setVertexBuffer(0, vb);
    pass.setBindGroup(0, bg);
    pass.draw(6);
    pass.end();
    d.queue.submit([enc.finish()]);

    return tex;
  }

  dispose() {
    this.envCubemap?.destroy();
    this.irradianceMap?.destroy();
    this.prefilterMap?.destroy();
    this.brdfLUT?.destroy();
    this.cubeVB?.destroy();
  }
}
