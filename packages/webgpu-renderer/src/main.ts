import './style.css';
import type { MaterialProps, Renderer, ScreenshotHooks } from '@sliced/shared';
import { loadSegbin } from '@sliced/shared';
import { SlicedPipeline } from './pipeline';
import { OrbitCamera } from './camera';
import { buildSegmentBuffers, HIDDEN_ROLES } from './buffer';

export class WebGPURenderer implements Renderer {
  device!: GPUDevice;
  context!: GPUCanvasContext;
  pipeline!: SlicedPipeline;
  camera!: OrbitCamera;
  firstFrame = true;
  disposeControls!: () => void;
  disposed = false;
  /** Readable stats for the overlay. Written each frame. */
  stats = { fps: 0, triangles: 0 };
  ssaoEnabled = true;
  /** Debug preview: show an internal texture instead of the normal composite. */
  debugPreview: 'none' | 'depth' | 'occlusion' | 'color' | 'shadow' = 'none';
  private _statsFrames = 0;
  private _statsTime = 0;

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
    const buffers = buildSegmentBuffers(this.device, parsed, HIDDEN_ROLES);
    this.pipeline.setSegments(buffers);

    // Fit camera to model bounding box (skip hidden roles)
    const g = parsed.geoms;
    const roles = parsed.roles;
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (let i = 0; i < parsed.count; i++) {
      if (HIDDEN_ROLES.has(roles[i])) continue;
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
    this.camera.target = new Float64Array([cx, cy, cz]);
    this.camera.alpha = Math.PI / 4;
    this.camera.beta = Math.PI / 4;

    // Compute radius to fit model in viewport (both axes)
    const bSphereR = Math.sqrt(
      (maxX - minX) ** 2 + (maxY - minY) ** 2 + (maxZ - minZ) ** 2
    ) / 2;
    const canvas = this.context.canvas as HTMLCanvasElement;
    const aspect = canvas.width > 0 && canvas.height > 0 ? canvas.width / canvas.height : 1;
    const vFovHalf = this.camera.fov / 2;
    const hFovHalf = Math.atan(Math.tan(vFovHalf) * aspect);
    const minFovHalf = Math.min(vFovHalf, hFovHalf);
    this.camera.radius = (bSphereR / Math.sin(minFovHalf)) * 1.15;
    this.camera.update(canvas.width / canvas.height);
    this.pipeline.writeCameraUBO(this.camera);

    // Use OrbitCamera to compute shadow view matrix (guaranteed correct convention)
    const maxDim = Math.max(maxX - minX, maxY - minY, maxZ - minZ) / 2;
    const ld = this.pipeline.lightDir;
    const ll = Math.sqrt(ld[0]*ld[0] + ld[1]*ld[1] + ld[2]*ld[2]);
    const ldx = ld[0]/ll, ldy = ld[1]/ll, ldz = ld[2]/ll;

    // Shadow camera opposite light direction
    const shadowRadius = maxDim * 5;
    // Build view matrix using camera.ts lookAt convention
    const v = new Float64Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ]);
    const p = new Float64Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ]);
    
    {
      // Build shadow camera lookAt matrix matching camera.ts exactly
      const tx = cx, ty = cy, tz = cz;
      const px = cx - ldx * shadowRadius;
      const py = cy - ldy * shadowRadius;
      const pz = cz - ldz * shadowRadius;
      
      // fwd = target - position
      let fx = tx - px, fy = ty - py, fz = tz - pz;
      const fLen = Math.sqrt(fx*fx + fy*fy + fz*fz);
      if (fLen > 0) { fx /= fLen; fy /= fLen; fz /= fLen; }
      
      // right = fwd × up (as in camera.ts)
      let rx = fy, ry = -fx, rz = 0;
      const rLen = Math.sqrt(rx*rx + ry*ry);
      if (rLen > 0.001) { rx /= rLen; ry /= rLen; }
      else { rx = 1; ry = 0; }
      
      // up = right × fwd (as in camera.ts)
      const ux = ry*fz - rz*fy;
      const uy = rz*fx - rx*fz;
      const uz = rx*fy - ry*fx;
      
      v[0] = rx;  v[4] = ux;  v[8]  = -fx;  v[12] = -(rx*px + ry*py + rz*pz);
      v[1] = ry;  v[5] = uy;  v[9]  = -fy;  v[13] = -(ux*px + uy*py + uz*pz);
      v[2] = rz;  v[6] = uz;  v[10] = -fz;  v[14] =  fx*px + fy*py + fz*pz;
      v[3] = 0;   v[7] = 0;   v[11] = 0;    v[15] = 1;
    }
    
    // Orthographic projection from bounding box corners
    let lmnX = Infinity, lmxX = -Infinity, lmnY = Infinity, lmxY = -Infinity;
    let lmnZ = Infinity, lmxZ = -Infinity;
    for (const sx of [minX, maxX]) {
      for (const sy of [minY, maxY]) {
        for (const sz of [minZ, maxZ]) {
          const x = v[0]*sx + v[4]*sy + v[8]*sz + v[12];
          const y = v[1]*sx + v[5]*sy + v[9]*sz + v[13];
          const z = v[2]*sx + v[6]*sy + v[10]*sz + v[14];
          if (x < lmnX) lmnX = x; if (x > lmxX) lmxX = x;
          if (y < lmnY) lmnY = y; if (y > lmxY) lmxY = y;
          if (z < lmnZ) lmnZ = z; if (z > lmxZ) lmxZ = z;
        }
      }
    }
    // Include camera at z=0
    if (0 < lmnZ) lmnZ = 0; if (0 > lmxZ) lmxZ = 0;
    
    const pad = 1.5;
    const hlw = Math.max(Math.abs(lmnX), Math.abs(lmxX)) * pad;
    const hlh = Math.max(Math.abs(lmnY), Math.abs(lmxY)) * pad;
    const zn = lmnZ - 0.5;   // near (closest, less negative)
    const zf = lmxZ + 0.5;   // far (farthest, more negative)
    const depthRange = zf - zn;
    
    p[0] = 1/hlw; p[4] = 0;    p[8]  = 0;          p[12] = 0;
    p[1] = 0;     p[5] = 1/hlh; p[9]  = 0;          p[13] = 0;
    p[2] = 0;     p[6] = 0;     p[10] = 1/depthRange; p[14] = -zn/depthRange;
    p[3] = 0;     p[7] = 0;     p[11] = 0;          p[15] = 1;
    
    // Multiply: shadowVP = p × v
    const svp = new Float32Array(16);
    for (let col = 0; col < 4; col++) {
      for (let row = 0; row < 4; row++) {
        svp[col * 4 + row] = p[row] * v[col * 4] + p[4 + row] * v[col * 4 + 1]
                           + p[8 + row] * v[col * 4 + 2] + p[12 + row] * v[col * 4 + 3];
      }
    }
    this.device.queue.writeBuffer(this.pipeline.shadowVPBuf, 0, svp);

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

    // Resize SSAO textures
    this.pipeline.resizeSSAO(w, h);

    // Update camera for new aspect ratio
    this.camera.update(w / h);
    this.pipeline.writeCameraUBO(this.camera);
  }

  dispose(): void {
    this.disposed = true;
    this.disposeControls?.();
    this.pipeline?.dispose();
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

    // Ensure SSAO textures are sized for current viewport
    this.pipeline.resizeSSAO(w, h);

    // LOD culling compute pass (submits its own encoder)
    this.pipeline.resetIndirect();
    this.pipeline.dispatchCull(h);

    const encoder = this.device.createCommandEncoder();

    // Shadow map render pass (always, regardless of SSAO)
    this.pipeline.renderShadowMap(encoder);

    // Offscreen render pass (scene → offscreen color + depth32float)
    {
      const offPass = encoder.beginRenderPass({
        colorAttachments: [{
          view: this.pipeline.offscreenColorTex.createView(),
          clearValue: { r: 0.15, g: 0.15, b: 0.17, a: 1.0 },
          loadOp: 'clear',
          storeOp: 'store',
        }],
        depthStencilAttachment: {
          view: this.pipeline.ssaoDepthTex.createView(),
          depthClearValue: 1.0,
          depthLoadOp: 'clear',
          depthStoreOp: 'store',
        },
      });
      this.pipeline.drawBody(offPass);
      this.pipeline.drawCaps(offPass);
      offPass.end();
    }

    // SSAO compute pass
    this.pipeline.dispatchSSAO(encoder);

    if (this.debugPreview !== 'none') {
      // Debug preview: render selected texture to swapchain (scene still renders above)
      const swapView = this.context.getCurrentTexture().createView();
      const swapPass = encoder.beginRenderPass({
        colorAttachments: [{
          view: swapView,
          loadOp: 'clear',
          storeOp: 'store',
        }],
      });
      this.pipeline.renderDebugView(swapPass, this.debugPreview);
      swapPass.end();
    } else if (this.ssaoEnabled) {
      {
        const swapView = this.context.getCurrentTexture().createView();
        const swapPass = encoder.beginRenderPass({
          colorAttachments: [{
            view: swapView,
            clearValue: { r: 0.15, g: 0.15, b: 0.17, a: 1.0 },
            loadOp: 'clear',
            storeOp: 'store',
          }],
        });
        this.pipeline.composite(swapPass);
        swapPass.end();
      }
    } else {
      // Render directly to swapchain (no SSAO)
      const swapView = this.context.getCurrentTexture().createView();
      const swapPass = encoder.beginRenderPass({
        colorAttachments: [{
          view: swapView,
          clearValue: { r: 0.15, g: 0.15, b: 0.17, a: 1.0 },
          loadOp: 'clear',
          storeOp: 'store',
        }],
        depthStencilAttachment: {
          view: this.pipeline.ssaoDepthTex.createView(),
          depthClearValue: 1.0,
          depthLoadOp: 'clear',
          depthStoreOp: 'store',
        },
      });
      this.pipeline.drawBody(swapPass);
      this.pipeline.drawCaps(swapPass);
      swapPass.end();
    }

    this.device.queue.submit([encoder.finish()]);

    // FPS tracking
    this._statsFrames++;
    const now = performance.now();
    if (now - this._statsTime >= 1000) {
      this.stats.fps = Math.round(this._statsFrames / ((now - this._statsTime) / 1000));
      this._statsFrames = 0;
      this._statsTime = now;

      // Read LOD counters for triangle count
      const rb = this.device.createBuffer({
        size: 12,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const re = this.device.createCommandEncoder();
      re.copyBufferToBuffer(this.pipeline.lodCountersBuf, 0, rb, 0, 12);
      this.device.queue.submit([re.finish()]);
      rb.mapAsync(GPUMapMode.READ).then(() => {
        const v = new Uint32Array(rb.getMappedRange());
        const tris = (this.pipeline.bodyIC[0] / 3) * v[0] +
                     (this.pipeline.bodyIC[1] / 3) * v[1] +
                     (this.pipeline.bodyIC[2] / 3) * v[2] +
                     (this.pipeline.capIC[0] / 3) * (this.pipeline.segmentBuffers?.capCount || 0);
        this.stats.triangles = Math.round(tris / 1000);
        rb.destroy();
      });
    }

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
