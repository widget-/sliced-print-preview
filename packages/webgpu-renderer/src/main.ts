import './style.css';
import type { MaterialProps, Renderer, ScreenshotHooks } from '@sliced/shared';
import { loadSegbin } from '@sliced/shared';
import { SlicedPipeline } from './pipeline';
import { OrbitCamera } from './camera';
import { buildSegmentBuffers } from './buffer';

export class WebGPURenderer implements Renderer {
  device!: GPUDevice;
  context!: GPUCanvasContext;
  pipeline!: SlicedPipeline;
  camera!: OrbitCamera;
  depthTexture!: GPUTexture;
  disposeControls!: () => void;
  disposed = false;

  async mount(_container: HTMLElement, canvas: HTMLCanvasElement): Promise<void> {
    if (!canvas) throw new Error('Canvas element is null — mount called before DOM ready');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No WebGPU adapter');
    const device = await adapter.requestDevice();
    this.device = device;

    const context = canvas.getContext('webgpu') as GPUCanvasContext;
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: 'premultiplied' });
    this.context = context;
    this.pipeline = new SlicedPipeline(device);
    await this.pipeline.init();

    this.camera = new OrbitCamera();
    this.disposeControls = this.camera.attach(canvas);

    this.resize();
    this.disposed = false;
    this._loop();
  }

  async loadModel(url: string): Promise<number> {
    const t0 = performance.now();
    const parsed = await loadSegbin(url);
    const buffers = buildSegmentBuffers(this.device, parsed);
    this.pipeline.setSegments(buffers);

    // Fit camera to model bounding box
    const g = parsed.geoms;
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (let i = 0; i < parsed.count; i++) {
      const sx = g[i * 8], sy = g[i * 8 + 1], sz = g[i * 8 + 2];
      const ex = g[i * 8 + 3], ey = g[i * 8 + 4], ez = g[i * 8 + 5];
      if (sx < minX) minX = sx; if (sx > maxX) maxX = sx;
      if (sy < minY) minY = sy; if (sy > maxY) maxY = sy;
      if (sz < minZ) minZ = sz; if (sz > maxZ) maxZ = sz;
      if (ex < minX) minX = ex; if (ex > maxX) maxX = ex;
      if (ey < minY) minY = ey; if (ey > maxY) maxY = ey;
      if (ez < minZ) minZ = ez; if (ez > maxZ) maxZ = ez;
    }
    const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2, cz = (minZ + maxZ) / 2;
    const maxDim = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 1);
    this.camera.target = new Float64Array([cx, cy, cz]);
    this.camera.radius = maxDim * 1.5;

    return Math.round(performance.now() - t0);
  }

  setMaterial(props: MaterialProps): void {
    this.pipeline.material.roughness = props.roughness;
    this.pipeline.material.metalness = props.metalness;
    this.pipeline.material.envIntensity = props.envIntensity;
    this.pipeline.material.specularStrength = props.specularStrength;
    this.pipeline.material.ambientStrength = props.ambientStrength;
    this.pipeline.material.baseColorTint = [
      parseInt(props.baseColorTint.slice(1, 3), 16) / 255,
      parseInt(props.baseColorTint.slice(3, 5), 16) / 255,
      parseInt(props.baseColorTint.slice(5, 7), 16) / 255,
    ];
    this.pipeline.writeMaterialUBO();
  }

  resize(): void {
    if (!this.context) return;
    const canvas = this.context.canvas as HTMLCanvasElement;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (w === 0 || h === 0) return;

    // Only resize if dimensions actually changed
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }

    // Recreate depth texture at new size
    if (this.depthTexture) this.depthTexture.destroy();
    this.depthTexture = this.device.createTexture({
      size: { width: w, height: h },
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }

  dispose(): void {
    this.disposed = true;
    this.disposeControls?.();
    this.pipeline?.dispose();
    this.depthTexture?.destroy();
    this.device?.destroy();
  }

  getScreenshotHooks(): ScreenshotHooks | null {
    return null;
  }

  // ── Private render loop ──

  private _loop = () => {
    if (this.disposed) return;
    const canvas = this.context.canvas as HTMLCanvasElement;
    const w = canvas.width;
    const h = canvas.height;
    if (w === 0 || h === 0) {
      requestAnimationFrame(this._loop);
      return;
    }

    // Update camera
    this.camera.update(w / h);
    this.pipeline.writeCameraUBO(this.camera);

    // Render
    const view = this.context.getCurrentTexture().createView();
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view,
        clearValue: { r: 0.15, g: 0.15, b: 0.17, a: 1.0 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
      depthStencilAttachment: {
        view: this.depthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    });
    this.pipeline.draw(pass);
    this.pipeline.drawCaps(pass);
    pass.end();
    this.device.queue.submit([encoder.finish()]);

    requestAnimationFrame(this._loop);
  };
}

// ── Standalone dev entry point ──
async function main() {
  const canvas = document.getElementById('render-canvas') as HTMLCanvasElement;
  const fallback = document.getElementById('fallback') as HTMLParagraphElement;

  if (!navigator.gpu) {
    fallback.style.display = 'block';
    canvas.style.display = 'none';
    return;
  }

  const renderer = new WebGPURenderer();
  await renderer.mount(document.getElementById('app')!, canvas);

  // Load a segbin model if provided via URL param
  const segbinParam = new URLSearchParams(location.search).get('segbin');
  if (segbinParam) {
    try {
      await renderer.loadModel(segbinParam);
      console.log('[WebGPU] Model loaded:', segbinParam);
    } catch (e) {
      console.error('[WebGPU] Failed to load model:', e);
    }
  }
}

// Only run standalone when the dev page's canvas element exists
if (typeof document !== 'undefined' && document.getElementById('render-canvas')) {
  main().catch(console.error);
}
