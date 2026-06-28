import type { MaterialProps, Renderer, ScreenshotHooks } from '@sliced/shared';
import { loadSegbin } from '@sliced/shared';
import { WebGLPipeline } from './pipeline';
import { OrbitCamera } from './camera';
import { buildSegmentTextures, HIDDEN_ROLES } from './buffer';

export class WebGLRenderer implements Renderer {
  gl!: WebGL2RenderingContext;
  pipeline!: WebGLPipeline;
  camera!: OrbitCamera;
  firstFrame = true;
  disposeControls!: () => void;
  disposed = false;
  stats = { fps: 0, triangles: 0 };
  ssaoEnabled = false; // SSAO not supported in WebGL fallback
  debugPreview: 'none' = 'none';

  private _mounted = false;
  private _loopActive = false;
  private _idleFrames = 0;
  private _prevCamPos = new Float64Array(3);
  private readonly IDLE_THRESHOLD = 20;
  private _lastFrameTime = 0;
  private _minFrameInterval = 1000 / 60;
  private _statsFrames = 0;
  private _statsTime = 0;
  private _canvas!: HTMLCanvasElement;

  async mount(_container: HTMLElement, canvas: HTMLCanvasElement): Promise<void> {
    if (!canvas) throw new Error('Canvas element is null');

    const gl = canvas.getContext('webgl2', {
      alpha: true,
      antialias: false,
      depth: true,
      premultipliedAlpha: false,
    });
    if (!gl) throw new Error('WebGL 2.0 not available');

    this.gl = gl;
    this._canvas = canvas;

    this.pipeline = new WebGLPipeline(gl);
    this.pipeline.init();

    // Load default env map
    try {
      await this.pipeline.setEnvMap('ferndale_studio_07_1k.hdr');
    } catch {
      // Env map is optional
    }

    this.camera = new OrbitCamera();
    this.camera.onInteraction = () => this._startLoop();
    this.disposeControls = this.camera.attach(canvas);

    this.disposed = false;
    this._mounted = true;
    this.resize();
    this._idleFrames = 0;
    this._startLoop();
  }

  async loadModel(url: string): Promise<number> {
    const t0 = performance.now();
    const parsed = await loadSegbin(url);
    const segTexs = buildSegmentTextures(this.gl, parsed, HIDDEN_ROLES);
    this.pipeline.setSegments(segTexs, parsed);

    // Fit camera to model bounding box
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

    const bSphereR = Math.sqrt(
      (maxX - minX) ** 2 + (maxY - minY) ** 2 + (maxZ - minZ) ** 2
    ) / 2;
    const w = this._canvas.width, h = this._canvas.height;
    const aspect = w > 0 && h > 0 ? w / h : 1;
    const vFovHalf = this.camera.fov / 2;
    const hFovHalf = Math.atan(Math.tan(vFovHalf) * aspect);
    this.camera.radius = (bSphereR / Math.sin(Math.min(vFovHalf, hFovHalf))) * 1.15;
    this.camera.update(aspect);
    this.pipeline.writeCameraUBO(this.camera);

    const maxDim = Math.max(maxX - minX, maxY - minY, maxZ - minZ) / 2;
    this.pipeline.setModelBounds(cx, cy, cz, maxDim, maxDim, maxDim);

    // Place ground just below the model
    this.pipeline.setGroundZ(minZ - 0.5);

    this._startLoop();
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
    this._startLoop();
  }

  setShadowSoftness(v: number) { this.pipeline.shadowSoftness = v; this._startLoop(); }
  setKeyLightIntensity(v: number) { this.pipeline.lightDir[3] = v; this.pipeline.writeLightDirUBO(); this._startLoop(); }
  setFillLightIntensity(v: number) { this.pipeline.lightDir2[3] = v; this.pipeline.writeLightDir2UBO(); this._startLoop(); }
  setContactShadowDist(_v: number) { /* unsupported */ }
  setContactShadowStrength(_v: number) { /* unsupported */ }
  setSSAOIntensity(_v: number) { /* unsupported */ }
  setSSAORadius(_v: number) { /* unsupported */ }

  async setEnvMap(url: string): Promise<void> {
    await this.pipeline.setEnvMap(url);
    this._startLoop();
  }

  resize(): void {
    if (!this.gl || !this._mounted) return;
    const canvas = this._canvas;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (w === 0 || h === 0) return;

    const iw = Math.round(w);
    const ih = Math.round(h);
    if (canvas.width !== iw || canvas.height !== ih) {
      canvas.width = iw;
      canvas.height = ih;
    }

    this.gl.viewport(0, 0, canvas.width, canvas.height);
    this.camera.update(canvas.width / canvas.height);
    this.pipeline.writeCameraUBO(this.camera);
  }

  dispose(): void {
    this.disposed = true;
    this.disposeControls?.();
    this.pipeline?.dispose();
  }

  getScreenshotHooks(): ScreenshotHooks | null {
    return null;
  }

  // ── Render loop ──

  private _startLoop() {
    this._idleFrames = 0;
    if (!this._loopActive && !this.disposed) {
      this._loopActive = true;
      requestAnimationFrame(this._loop);
    }
  }

  private _loop = () => {
    if (this.disposed) { this._loopActive = false; return; }
    const now = performance.now();
    if (now - this._lastFrameTime < this._minFrameInterval && this._lastFrameTime > 0) {
      requestAnimationFrame(this._loop);
      return;
    }
    this._lastFrameTime = now;

    const canvas = this._canvas;
    const w = canvas.width;
    const h = canvas.height;
    if (w === 0 || h === 0) {
      requestAnimationFrame(this._loop);
      return;
    }

    this.camera.update(w / h);
    this.pipeline.writeCameraUBO(this.camera);

    const gl = this.gl;

    // Only render if a model is loaded
    if (this.pipeline.segmentTextures) {
      // 1. Shadow pass
      gl.enable(gl.DEPTH_TEST);
      gl.depthFunc(gl.LEQUAL);
      gl.depthMask(true);
      this.pipeline.renderShadowMap(this.camera, w, h);

      // 1b. Second light shadow pass
      this.pipeline.renderShadowMap2(this.camera, w, h);

      // 2. Main pass (direct to canvas)
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, w, h);
      gl.clearColor(0.15, 0.15, 0.17, 1.0);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST);
      gl.depthFunc(gl.LEQUAL);
      gl.depthMask(true);

      this.pipeline.drawBody(this.camera, canvas);
      this.pipeline.drawCaps();
      this.pipeline.drawGround();
    }

    // FPS tracking
    this._statsFrames++;
    const now2 = performance.now();
    if (now2 - this._statsTime >= 1000) {
      this.stats.fps = Math.round(this._statsFrames / ((now2 - this._statsTime) / 1000));
      this._statsFrames = 0;
      this._statsTime = now2;

      // Triangle count from actual LOD distribution
      const [c0, c1, c2] = this.pipeline._lodCounts;
      const tris =
        (this.pipeline.bodyIC[0] / 3) * c0 +
        (this.pipeline.bodyIC[1] / 3) * c1 +
        (this.pipeline.bodyIC[2] / 3) * c2 +
        (this.pipeline.capIC[0] / 3) * (this.pipeline.segmentTextures?.capCount || 0);
      this.stats.triangles = Math.round(tris / 1000);
    }

    // Idle detection
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
    }
  };
}

// ── Standalone dev entry point ──
async function main() {
  const canvas = document.getElementById('render-canvas') as HTMLCanvasElement;
  const fallback = document.getElementById('fallback') as HTMLParagraphElement;

  // Check WebGL 2.0 support
  const testCanvas = document.createElement('canvas');
  const testGl = testCanvas.getContext('webgl2');
  if (!testGl) {
    fallback.style.display = 'block';
    canvas.style.display = 'none';
    return;
  }

  const renderer = new WebGLRenderer();
  await renderer.mount(document.getElementById('app')!, canvas);

  const segbinParam = new URLSearchParams(location.search).get('segbin');
  if (segbinParam) {
    try {
      await renderer.loadModel(segbinParam);
      console.log('[WebGL2] Model loaded:', segbinParam);
    } catch (e) {
      console.error('[WebGL2] Failed to load model:', e);
    }
  }
}

if (typeof document !== 'undefined' && document.getElementById('render-canvas')) {
  main().catch(console.error);
}
