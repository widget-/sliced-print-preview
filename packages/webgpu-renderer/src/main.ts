import './style.css';
import './console-relay';
import type { MaterialProps, Renderer, ScreenshotHooks } from '@sliced/shared';
import { loadSegbin } from '@sliced/shared';
import { SlicedPipeline } from './pipeline';
import { OrbitCamera } from './camera';
import { mat4Inverse } from './math';
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
  debugPreview: string = 'none';
  private _mounted = false;
  private _statsFrames = 0;
  private _statsTime = 0;
  private _debugFrames = 0;
  private _lastFrameTime = 0;
  private _timingAfterLoad = 0;
  private _minFrameInterval = 1000 / 60;
  private _querySet?: GPUQuerySet;
  private _queryResolveBuf?: GPUBuffer;
  private _queryResultBuf?: GPUBuffer;
  private _gpuTiming = false;

  async mount(_container: HTMLElement, canvas: HTMLCanvasElement): Promise<void> {
    if (!canvas) throw new Error('Canvas element is null — mount called before DOM ready');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No WebGPU adapter');
    const hasTimestamp = adapter.features?.has?.('timestamp-query');
    if (hasTimestamp) console.log('✅ GPU timestamp-query supported');
    const device = await adapter.requestDevice({
      requiredFeatures: hasTimestamp ? ['timestamp-query'] : [],
    });
    this.device = device;
    // Relay WebGPU validation errors to the dev server console
    ;(window as any).__relayGPUError?.(device);

    // Set up GPU timestamp queries (requires chrome://flags/#enable-webgpu-developer-features)
    if (hasTimestamp) {
      const qCount = 8;
      this._querySet = device.createQuerySet({ type: 'timestamp', count: qCount });
      this._queryResolveBuf = device.createBuffer({
        size: qCount * 8,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
      this._queryResultBuf = device.createBuffer({
        size: qCount * 8,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      this._gpuTiming = true;
    }

    const context = canvas.getContext('webgpu') as GPUCanvasContext;
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: 'premultiplied' });
    this.context = context;
    this.pipeline = new SlicedPipeline(device);
    await this.pipeline.init();
    this.pipeline.taaEnabled = true;

    this.camera = new OrbitCamera();
    this.camera.onInteraction = () => this._startLoop();
    this.pipeline.setSSAOCamera(this.camera.near, this.camera.far, this.camera.fov);
    this.disposeControls = this.camera.attach(canvas);

    this.disposed = false;
    this._mounted = true;
    this.resize();
    this._idleFrames = 0;
    this._startLoop();

    // Resume render loop when user returns to the tab
    document.addEventListener('visibilitychange', () => {
      if (!document.hidden && !this._loopActive && !this.disposed) {
        this._startLoop();
      }
    });
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

    // Place ground just below the model
    this.pipeline.setGroundZ(minZ - 0.5);

    // Use OrbitCamera to compute shadow view matrix (guaranteed correct convention)
    const maxDim = Math.max(maxX - minX, maxY - minY, maxZ - minZ) / 2;

    // Helper to compute a shadow VP matrix for a given light direction
    // Store model bounds and compute shadow VP for both lights
    this.pipeline.setModelBounds(cx, cy, cz, maxDim, maxDim, maxDim);
    this.pipeline.writeShadowVP(this.pipeline.lightDir[0], this.pipeline.lightDir[1], this.pipeline.lightDir[2], this.pipeline.shadowVPBuf);
    this.pipeline.writeShadowVP(this.pipeline.lightDir2[0], this.pipeline.lightDir2[1], this.pipeline.lightDir2[2], this.pipeline.shadowVPBuf2);

    this.triggerGPUTiming();
    this._startLoop();
    return Math.round(performance.now() - t0);
  }

  /** Schedule a GPU timing capture on the second frame after this call. */
  triggerGPUTiming() { this._timingAfterLoad = this._debugFrames; }

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
    this._startLoop();
  }

  setShadowSoftness(v: number) { this.pipeline.shadowSoftness = v; this.pipeline.writeShadowSoftness(); this._startLoop(); }

  setKeyLightIntensity(v: number) { this.pipeline.lightDir[3] = v; this.pipeline.writeLightDirUBO(); this._startLoop(); }

  setFillLightIntensity(v: number) { this.pipeline.lightDir2[3] = v; this.pipeline.writeLightDir2UBO(); this._startLoop(); }

  setContactShadowDist(v: number) { this.pipeline.contactShadowDist = v; this._startLoop(); }

  setContactShadowStrength(v: number) { this.pipeline.contactShadowStrength = v; this._startLoop(); }

  setSSAOIntensity(v: number) { this.pipeline.ssaoIntensity = v; this.pipeline.writeSSAOParams(); this._startLoop(); }

  setSSAORadius(v: number) { this.pipeline.ssaoRadius = v; this.pipeline.writeSSAOParams(); this._startLoop(); }

  setArcCurvature(_v: number) {
    // Arc curvature is a PlayCanvas-specific feature (conic fillets).
    // WebGPU renderer handles arcs differently; this is a no-op.
  }

  async setEnvMap(url: string): Promise<void> {
    await this.pipeline.setEnvMap(url);
    this._startLoop();
  }

  resize(): void {
    if (!this.context || !this._mounted) return;
    const canvas = this.context.canvas as HTMLCanvasElement;
    const cssW = canvas.clientWidth;
    const cssH = canvas.clientHeight;
    if (cssW === 0 || cssH === 0) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const physW = Math.round(cssW * dpr);
    const physH = Math.round(cssH * dpr);

    // Only resize if physical dimensions actually changed
    if (canvas.width !== physW || canvas.height !== physH) {
      canvas.width = physW;
      canvas.height = physH;
    }
    // Keep CSS size at logical pixels so the canvas isn't shrunk by the buffer
    canvas.style.width = `${cssW}px`;
    canvas.style.height = `${cssH}px`;

    // Resize SSAO textures using physical pixel size
    this.pipeline.resizeSSAO(canvas.width, canvas.height);

    // Update camera for new aspect ratio
    this.camera.update(canvas.width / canvas.height);
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

  private _loopActive = false;
  private _idleFrames = 0;
  private _prevCamPos = new Float64Array(3);
  private readonly IDLE_THRESHOLD = 5; // stop after 5 stable frames

  /** Restart the render loop (call when camera or state changes). */
  private _startLoop() {
    this._idleFrames = 0;
    if (!this._loopActive && !this.disposed) {
      this._loopActive = true;
      requestAnimationFrame(this._loop);
    }
  }

  private _loop = () => {
    if (this.disposed) { this._loopActive = false; return; }
    // Pause when page is hidden (background tab / locked mobile screen)
    if (document.hidden) {
      this._loopActive = false;
      return;
    }
    const now = performance.now();
    if (now - this._lastFrameTime < this._minFrameInterval && this._lastFrameTime > 0) {
      requestAnimationFrame(this._loop);
      return;
    }
    if (this._lastFrameTime > 0) {
      const dt = now - this._lastFrameTime;
      const isTimingFrame = this._debugFrames === 1 || (this._timingAfterLoad > 0 && this._debugFrames === this._timingAfterLoad + 1);
      if (isTimingFrame) console.log(`frame interval: ${dt.toFixed(2)} ms (${(1000/dt).toFixed(1)} FPS)`);
    }
    this._lastFrameTime = now;
    this._debugFrames++;
    const canvas = this.context.canvas as HTMLCanvasElement;
    const w = canvas.width;
    const h = canvas.height;
    if (w === 0 || h === 0) {
      requestAnimationFrame(this._loop);
      return;
    }

    // Update camera with TAA jitter
    this.camera.update(w / h);

    // Write clean (unjittered) projection for SSAO hemisphere sampling
    // Must be done before jitter modifies camera.proj in-place below.
    this.device.queue.writeBuffer(this.pipeline.ssaoProjBuf, 0, new Float32Array(this.camera.proj));

    // Contact shadow also needs unjittered viewProj — save it before TAA jitter
    const cleanViewProj = new Float32Array(this.camera.viewProj);
    const cleanInvViewProj = new Float32Array(this.camera.invViewProj);

    // Apply Halton jitter to the projection for TAA
    if (this.pipeline.taaEnabled) {
      const [jx, jy] = SlicedPipeline.getTAAJitter(this.pipeline.taaFrame);
      SlicedPipeline.jitterProj(this.camera.proj, jx, jy, w, h);
      // Recompute viewProj with jittered projection
      const vp = new Float32Array(16);
      for (let col = 0; col < 4; col++) {
        for (let row = 0; row < 4; row++) {
          vp[col * 4 + row] =
            this.camera.proj[row] * this.camera.viewMat[col * 4] +
            this.camera.proj[4 + row] * this.camera.viewMat[col * 4 + 1] +
            this.camera.proj[8 + row] * this.camera.viewMat[col * 4 + 2] +
            this.camera.proj[12 + row] * this.camera.viewMat[col * 4 + 3];
        }
      }
      this.camera.viewProj.set(vp);
      mat4Inverse(this.camera.viewProj, this.camera.invViewProj);
    }
    this.pipeline.writeCameraUBO(this.camera);
    this.pipeline.writeContactShadow(cleanViewProj, cleanInvViewProj, this.pipeline.lightDir);

    // LOD culling compute pass (submits its own encoder)
    this.pipeline.resetIndirect();
    this.pipeline.dispatchCull(h);

    const encoder = this.device.createCommandEncoder();

    // Shadow map render pass (always, regardless of SSAO)
    // Recompute shadow VPs each frame so changes to lightDir take effect
    this.pipeline.writeShadowVP(this.pipeline.lightDir[0], this.pipeline.lightDir[1], this.pipeline.lightDir[2], this.pipeline.shadowVPBuf);
    this.pipeline.writeShadowVP(this.pipeline.lightDir2[0], this.pipeline.lightDir2[1], this.pipeline.lightDir2[2], this.pipeline.shadowVPBuf2);
    this.pipeline.renderShadowMap(encoder);
    this.pipeline.renderShadowMap2(encoder);

    if (this._gpuTiming) try { (encoder as any).writeTimestamp(this._querySet!, 0); } catch(e) { this._gpuTiming = false; }

    // Offscreen render pass (scene → offscreen color + depth32float)
    {
      const offPass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: this.pipeline.offscreenColorTex.createView(),
            clearValue: { r: 0.15, g: 0.15, b: 0.17, a: 1.0 },
            loadOp: 'clear',
            storeOp: 'store',
          },
          {
            view: this.pipeline.normalTex.createView(),
            clearValue: { r: 0.5, g: 0.5, b: 0.5, a: 1.0 }, // unpacked normal (0,0,1) → packed 0.5
            loadOp: 'clear',
            storeOp: 'store',
          },
        ],
        depthStencilAttachment: {
          view: this.pipeline.ssaoDepthTex.createView(),
          depthClearValue: 1.0,
          depthLoadOp: 'clear',
          depthStoreOp: 'store',
        },
      });
      this.pipeline.drawBody(offPass);
      this.pipeline.drawCaps(offPass);
      this.pipeline.drawGround(offPass);
      offPass.end();
    }

    // Screen-space contact shadows (always, needs depth from offscreen pass)
    this.pipeline.renderContactShadow(encoder);

    // SSAO compute pass + blur (only when enabled)
    if (this.ssaoEnabled) {
      if (this._gpuTiming) try { (encoder as any).writeTimestamp(this._querySet!, 1); } catch(e) {}
      this.pipeline.dispatchSSAO(encoder);
      if (this._gpuTiming) try { (encoder as any).writeTimestamp(this._querySet!, 2); } catch(e) {}
      this.pipeline.dispatchBlur(encoder);
      if (this._gpuTiming) try { (encoder as any).writeTimestamp(this._querySet!, 3); } catch(e) {}
    }

    // ── Velocity pass (screen-space motion) ──
    // Always compute velocity when TAA is enabled — needs depth from offscreen pass
    if (this.pipeline.taaEnabled) {
      this.pipeline.dispatchVelocity(encoder);
    }

    // ── Present to swapchain ──
    const sv = this.context.getCurrentTexture().createView();
    if (this.debugPreview !== 'none') {
      const swapPass = encoder.beginRenderPass({
        colorAttachments: [{ view: sv, loadOp: 'clear', storeOp: 'store' }],
      });
      this.pipeline.renderDebugView(swapPass, this.debugPreview);
      swapPass.end();
    } else if (this.pipeline.taaEnabled) {
      // TAA path:
      // 1. Composite offscreen color × occlusion → compositeTex
      // 2. TAA resolve: blend compositeTex + velocityTex + historyTex → swapchain + new history
      const compPass = encoder.beginRenderPass({
        colorAttachments: [{
          view: this.pipeline.compositeTex.createView(),
          loadOp: 'clear', storeOp: 'store',
        }],
      });
      this.pipeline.composite(compPass);
      compPass.end();

      this.pipeline.dispatchTAA(encoder, sv);
    } else if (this.ssaoEnabled) {
      // Composite: offscreen color × occlusion → swapchain
      const swapPass = encoder.beginRenderPass({
        colorAttachments: [{
          view: sv,
          clearValue: { r: 0.15, g: 0.15, b: 0.17, a: 1.0 },
          loadOp: 'clear', storeOp: 'store',
        }],
      });
      this.pipeline.composite(swapPass);
      swapPass.end();
    } else {
      // Copy offscreen color → swapchain (no SSAO)
      const swapPass = encoder.beginRenderPass({
        colorAttachments: [{
          view: sv,
          clearValue: { r: 0.15, g: 0.15, b: 0.17, a: 1.0 },
          loadOp: 'clear', storeOp: 'store',
        }],
      });
      this.pipeline.copyToSwapchain(swapPass);
      swapPass.end();
    }
    if (this._gpuTiming) try { (encoder as any).writeTimestamp(this._querySet!, 6); } catch(e) {}

    this.device.queue.submit([encoder.finish()]);

    // Read back GPU timestamps on second frame after mount or after model load
    const doTiming = this._debugFrames === 1 || (this._timingAfterLoad > 0 && this._debugFrames === this._timingAfterLoad + 1);
    if (this._gpuTiming && doTiming) {
      const qs = this._querySet!;
      const rb = this._queryResolveBuf!;
      const rb2 = this._queryResultBuf!;
      const re = this.device.createCommandEncoder();
      re.resolveQuerySet(qs, 0, 7, rb, 0);
      re.copyBufferToBuffer(rb, 0, rb2, 0, 7 * 8);
      this.device.queue.submit([re.finish()]);
      rb2.mapAsync(GPUMapMode.READ).then(() => {
        const arr = new BigInt64Array(rb2.getMappedRange());
        const labels = ['offscreen', 'ssao-start', 'blur-start', 'velocity-start', 'velocity-end', 'frame-end'];
        for (let i = 0; i < 5; i++) {
          if (arr[i] === 0n || arr[i+1] === 0n) continue;
          const ns = Number(arr[i+1] - arr[i]);
          console.log(`  GPU [${labels[i]}→${labels[i+1]}]: ${(ns / 1e6).toFixed(3)} ms`);
        }
        rb2.unmap();
      });
    }

    // FPS tracking (skip the expensive readback every frame — do it every 60 frames)
    this._statsFrames++;
    const now2 = performance.now();
    if (now2 - this._statsTime >= 1000) {
      this.stats.fps = Math.round(this._statsFrames / ((now2 - this._statsTime) / 1000));
      this._statsFrames = 0;
      this._statsTime = now2;

      // Read LOD counters for triangle count
      const rb = this.device.createBuffer({
        size: 12,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const re = this.device.createCommandEncoder();
      re.copyBufferToBuffer(this.pipeline.lodCountersBuf, 0, rb, 0, 12);
      if (this._timingAfterLoad > 0 && this._debugFrames === this._timingAfterLoad + 1) {
        console.log('--- GPU timings after model load ---');
      }
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

    // Idle detection: if the camera hasn't moved for enough frames, stop rendering.
    // The camera's onInteraction callback restarts the loop on any user input.
    const cam = this.camera;
    const moved = cam.position[0] !== this._prevCamPos[0]
               || cam.position[1] !== this._prevCamPos[1]
               || cam.position[2] !== this._prevCamPos[2];
    this._prevCamPos.set(cam.position);
    if (moved) {
      this._idleFrames = 0;
    } else {
      this._idleFrames++;
    }

    if (this._idleFrames < this.IDLE_THRESHOLD) {
      requestAnimationFrame(this._loop);
    } else {
      this._loopActive = false;
      // Free last frame's command encoder resources
      this.device.queue.onSubmittedWorkDone?.();
    }
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
