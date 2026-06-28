import type { MaterialProps, Renderer, ScreenshotHooks } from '@sliced/shared';
import { loadSegbin, type SegbinData } from '@sliced/shared';
import { generateBodyGeometry } from './geometry';
import * as pc from 'playcanvas';

// ── Segment data per-instance layout ──
// ATTR12: startPos.xyz + width          (vec4)
// ATTR13: endPos.xyz + flags            (vec4)  — flags.x: isArc
// ATTR14: color.rgb + _unused           (vec4)
const SEG_FLOATS = 12;

function packSegmentData(data: SegbinData): Float32Array {
  // Count visible segments (exclude SkirtBrim=10, Other=13)
  const n = data.count;
  let visCount = 0;
  for (let i = 0; i < n; i++) {
    const r = data.roles[i];
    if (r !== 10 && r !== 13) visCount++;
  }

  const out = new Float32Array(visCount * SEG_FLOATS);
  const g = data.geoms;
  let vi = 0;
  for (let i = 0; i < n; i++) {
    const r = data.roles[i];
    if (r === 10 || r === 13) continue; // skip SkirtBrim, Other

    const off = vi * SEG_FLOATS;
    // ATTR12: startPos.xyz + width  (rotate Z‑up → Y‑up: (x,y,z) → (x,z,-y))
    out[off + 0] = g[i * 8 + 0];      // x → x
    out[off + 1] = g[i * 8 + 2];      // z → y
    out[off + 2] = -g[i * 8 + 1];     // -y → z
    out[off + 3] = g[i * 8 + 6];      // width
    // ATTR13: endPos.xyz + flags
    out[off + 4] = g[i * 8 + 3];      // x → x
    out[off + 5] = g[i * 8 + 5];      // z → y
    out[off + 6] = -g[i * 8 + 4];     // -y → z
    const isArc = (data.segType[i] & 1) !== 0 ? 1.0 : 0.0;
    out[off + 7] = isArc;
    // ATTR14: color.rgb + unused
    const c = roleColorVec3(r);
    out[off + 8] = c[0];
    out[off + 9] = c[1];
    out[off + 10] = c[2];
    out[off + 11] = 0.0;
    vi++;
  }
  return out;
}

function roleColorVec3(r: number): [number, number, number] {
  const p: Record<number, [number, number, number]> = {
    0: [0.129, 0.286, 0.620], 1: [0.129, 0.286, 0.620],
    2: [0.882, 0.882, 0.882], 3: [0.941, 1.000, 0.424],
    4: [0.682, 0.620, 0.251], 5: [0.722, 0.188, 0.235],
    6: [1.000, 0.647, 0.008], 7: [0.180, 0.835, 0.451],
    8: [0.482, 0.929, 0.624], 9: [0.000, 0.000, 0.000],
    10: [0.643, 0.690, 0.745], 11: [0.200, 0.224, 0.235],
    12: [0.973, 0.761, 0.365], 13: [0.969, 0.000, 1.000],
  };
  return p[r] ?? [0.8, 0.8, 0.8];
}

function createInstancingFormat(device: pc.GraphicsDevice): pc.VertexFormat {
  return new pc.VertexFormat(device, [
    { semantic: pc.SEMANTIC_ATTR12, components: 4, type: pc.TYPE_FLOAT32 },
    { semantic: pc.SEMANTIC_ATTR13, components: 4, type: pc.TYPE_FLOAT32 },
    { semantic: pc.SEMANTIC_ATTR14, components: 4, type: pc.TYPE_FLOAT32 },
  ]);
}

// ── GLSL vertex shader (WebGL2) ──
const BODY_VERT_GLSL = `
attribute vec3 vertex_position;
attribute vec3 vertex_normal;

#ifdef INSTANCING
attribute vec4 instance_line1;
attribute vec4 instance_line2;
attribute vec4 instance_line3;
#endif

uniform mat4 matrix_viewProjection;
uniform mat4 matrix_view;
uniform vec3 matrix_viewPosition;
uniform vec3 lightDir;
uniform float lightIntensity;
uniform float ambientStrength;

varying vec3 vWorldPos;
varying vec3 vWorldNormal;
varying vec3 vColor;
varying vec4 vClipPos;

const float hScale = 1.25;
const float areaCorrection = 1.1;
const vec3 upDirConst = vec3(0.0, 1.0, 0.0); // PlayCanvas Y‑up

mat4 getModelMatrix() {
  // Not used — we compute world pos directly
  return mat4(1.0);
}

void main() {
#ifdef INSTANCING
  vec3 startPos = instance_line1.xyz;
  float width = instance_line1.w;
  vec3 endPos = instance_line2.xyz;
  float flags = instance_line2.w;
  bool isArc = flags > 0.5;
  vec3 segColor = instance_line3.rgb;
#else
  // Fallback: draw a single extruded segment at origin for debugging
  vec3 startPos = vec3(-0.5, 0.0, 0.0);
  float width = 0.4;
  vec3 endPos = vec3(0.5, 0.0, 0.0);
  bool isArc = false;
  vec3 segColor = vec3(1.0, 0.5, 0.2);
#endif

  float t = vertex_position.z + 0.5;

  vec3 segPos;
  vec3 endTangent;

  if (isArc) {
    // Rational quadratic Bezier approximation using next segment
    // For now: linear fallback (full Bezier needs neighbor data)
    segPos = mix(startPos, endPos, t);
    vec3 dir = endPos - startPos;
    float len = length(dir);
    endTangent = len > 0.001 ? dir / len : vec3(0.0, 0.0, 1.0);
  } else {
    segPos = mix(startPos, endPos, t);
    vec3 dir = endPos - startPos;
    float len = length(dir);
    endTangent = len > 0.001 ? dir / len : vec3(0.0, 0.0, 1.0);
  }

  vec3 tangent = endTangent;
  vec3 upDir = upDirConst;
  vec3 rightDir = -normalize(cross(upDir, tangent));
  if (length(rightDir) < 0.001) rightDir = vec3(1.0, 0.0, 0.0);
  vec3 fwdDir = -normalize(cross(rightDir, upDir));
  mat3 rot = mat3(rightDir, upDir, fwdDir);

  vec3 local = vec3(
    vertex_position.x * width * areaCorrection,
    vertex_position.y * width * hScale,
    0.0
  );

  vec3 worldPos = segPos + rot * local;
  vec3 worldNormal = normalize(rot * vertex_normal);

  gl_Position = matrix_viewProjection * vec4(worldPos, 1.0);
  vWorldPos = worldPos;
  vWorldNormal = worldNormal;
  vColor = segColor;
  vClipPos = gl_Position;
}
`;

// ── GLSL fragment shader — simple directional light ──
const BODY_FRAG_GLSL = `
varying vec3 vWorldPos;
varying vec3 vWorldNormal;
varying vec3 vColor;
varying vec4 vClipPos;

uniform vec3 matrix_viewPosition;
uniform vec3 lightDir;
uniform float lightIntensity;
uniform float ambientStrength;

void main() {
  vec3 N = normalize(vWorldNormal);
  vec3 L = normalize(lightDir);
  vec3 V = normalize(matrix_viewPosition - vWorldPos);
  vec3 H = normalize(L + V);

  float NdotL = max(dot(N, L), 0.0);
  float NdotH = max(dot(N, H), 0.0);
  float spec = pow(NdotH, 32.0);

  vec3 ambient = vColor * ambientStrength;
  vec3 diffuse = vColor * NdotL * lightIntensity;
  vec3 specular = vec3(1.0) * spec * lightIntensity * 0.5;

  vec3 result = ambient + diffuse + specular;
  gl_FragColor = vec4(result, 1.0);
}
`;

export class PlayCanvasRenderer implements Renderer {
  app!: pc.Application;
  private _canvas?: HTMLCanvasElement;
  private _container?: HTMLElement;
  private _meshEntity?: pc.Entity;
  private _cameraEntity?: pc.Entity;
  private _disposed = false;
  private _resizeObserver?: ResizeObserver;
  private _statsTimer?: ReturnType<typeof setInterval>;
  private _fpsFrames = 0;
  private _fpsTime = 0;
  private _material?: pc.ShaderMaterial;

  stats = { fps: 0, triangles: 0 };

  async mount(container: HTMLElement, canvas: HTMLCanvasElement): Promise<void> {
    this._canvas = canvas;
    this._container = container;

    this.app = new pc.Application(canvas, {
      graphicsDeviceOptions: {
        antialias: true,
        alpha: false,
        preferWebGl2: true,
        powerPreference: 'high-performance',
      },
    });

    const app = this.app;

    // Disable auto-render — we control the loop
    app.autoRender = false;

    this.resize();

    // ── Camera ──
    this._cameraEntity = new pc.Entity('camera');
    this._cameraEntity.addComponent('camera', {
      clearColor: new pc.Color(0.1, 0.1, 0.15, 1),
      fov: 60,
      nearClip: 0.1,
      farClip: 1000,
    });
    app.root.addChild(this._cameraEntity);

    // Debug: bright clear to confirm rendering
    console.log('[PlayCanvas] Camera entity created, clearColor:', this._cameraEntity.camera?.clearColor);

    // ── Lights ──
    const keyLight = new pc.Entity('keyLight');
    keyLight.addComponent('light', {
      type: 'directional',
      color: new pc.Color(1.0, 0.95, 0.85),
      intensity: 1.0,
      castShadows: true,
      shadowResolution: 1024,
      shadowDistance: 500,
      numCascades: 1,
    });
    keyLight.setLocalEulerAngles(45, 30, 0);
    app.root.addChild(keyLight);

    const fillLight = new pc.Entity('fillLight');
    fillLight.addComponent('light', {
      type: 'directional',
      color: new pc.Color(0.6, 0.7, 0.9),
      intensity: 0.4,
      castShadows: true,
      shadowResolution: 1024,
      shadowDistance: 500,
      numCascades: 1,
    });
    fillLight.setLocalEulerAngles(-30, -45, 0);
    app.root.addChild(fillLight);

    app.scene.ambientLight = new pc.Color(0.3, 0.3, 0.35);

    // ── Skybox placeholder ──
    app.scene.envAtlas = null;

    // ── Orbit controls ──
    this._setupOrbitControls(canvas);

    // ── Resize ──
    this._resizeObserver = new ResizeObserver(() => this.resize());
    this._resizeObserver.observe(container);

    // ── Stats ──
    this._fpsTime = performance.now();
    this._statsTimer = setInterval(() => {
      const now = performance.now();
      const dt = (now - this._fpsTime) / 1000;
      this.stats.fps = Math.round(this._fpsFrames / dt);
      this._fpsFrames = 0;
      this._fpsTime = now;
    }, 1000);

    this._disposed = false;
    this._startLoop();
  }

  async loadModel(url: string): Promise<number> {
    const t0 = performance.now();
    const data = await loadSegbin(url);

    // Remove existing mesh
    if (this._meshEntity) {
      this._meshEntity.destroy();
      this._meshEntity = undefined;
    }

    const device = this.app.graphicsDevice;

    // ── Build cross-section geometry mesh ──
    const bodyGeo = generateBodyGeometry(0.35, 0.35, 5, 3);
    const ringVertCount = bodyGeo.ringLen * bodyGeo.totalRings;

    // Create interleaved vertex buffer (position + normal)
    const interleavedData = new Float32Array(ringVertCount * 6);
    for (let i = 0; i < ringVertCount; i++) {
      const srcOff = i * 6;
      interleavedData[srcOff + 0] = bodyGeo.interleaved[srcOff + 0];
      interleavedData[srcOff + 1] = bodyGeo.interleaved[srcOff + 1];
      interleavedData[srcOff + 2] = bodyGeo.interleaved[srcOff + 2];
      interleavedData[srcOff + 3] = bodyGeo.interleaved[srcOff + 3];
      interleavedData[srcOff + 4] = bodyGeo.interleaved[srcOff + 4];
      interleavedData[srcOff + 5] = bodyGeo.interleaved[srcOff + 5];
    }

    const interleavedFormat = new pc.VertexFormat(device, [
      { semantic: pc.SEMANTIC_POSITION, components: 3, type: pc.TYPE_FLOAT32 },
      { semantic: pc.SEMANTIC_NORMAL, components: 3, type: pc.TYPE_FLOAT32 },
    ]);

    // For interleaved data: pack as a single buffer with stride = 24 bytes
    const finalMeshVb = new pc.VertexBuffer(device, interleavedFormat, ringVertCount, {
      data: interleavedData.buffer as ArrayBuffer,
    });

    const ib = new pc.IndexBuffer(device, pc.INDEXFORMAT_UINT16, bodyGeo.indices.length, pc.BUFFER_STATIC, bodyGeo.indices.buffer as ArrayBuffer);

    // Create mesh
    const mesh = new pc.Mesh(device);
    mesh.vertexBuffer = finalMeshVb;
    mesh.indexBuffer = [ib];
    mesh.primitive = [{
      type: pc.PRIMITIVE_TRIANGLES,
      base: 0,
      baseVertex: 0,
      count: bodyGeo.indices.length,
      indexed: true,
    }];
    mesh.update();

    // Create shader material (must exist before MeshInstance)
    // Debug: use simple passthrough shader first to verify mesh rendering
    const DEBUG_SHADER = false;
    const vertSrc = DEBUG_SHADER ? `
attribute vec3 vertex_position;
uniform mat4 matrix_viewProjection;
varying vec3 vColor;
void main() {
  gl_Position = matrix_viewProjection * vec4(vertex_position * 50.0, 1.0);
  vColor = vec3(1.0, 0.5, 0.0);
}` : BODY_VERT_GLSL;

    const fragSrc = DEBUG_SHADER ? `
varying vec3 vColor;
void main() {
  gl_FragColor = vec4(vColor, 1.0);
}` : BODY_FRAG_GLSL;

    const shaderDesc = {
      uniqueName: 'segment-body',
      vertexGLSL: vertSrc,
      fragmentGLSL: fragSrc,
      attributes: DEBUG_SHADER ? {
        vertex_position: pc.SEMANTIC_POSITION,
      } : {
        vertex_position: pc.SEMANTIC_POSITION,
        vertex_normal: pc.SEMANTIC_NORMAL,
        instance_line1: pc.SEMANTIC_ATTR12,
        instance_line2: pc.SEMANTIC_ATTR13,
        instance_line3: pc.SEMANTIC_ATTR14,
      },
    };
    this._material = new pc.ShaderMaterial(shaderDesc);

    // Disable instancing temporarily for debug shader
    // (instancing attributes not declared in debug shader)

    // Disable culling for debugging
    this._material.cull = pc.CULLFACE_NONE;

    if (!DEBUG_SHADER) {
      // Enable instancing define for ShaderMaterial
      this._material.defines.set('INSTANCING', '');
    }

    // Set light uniforms on the material (only for non-debug shader)
    if (!DEBUG_SHADER) {
      this._material.setParameter('lightDir', new Float32Array([1.0, 1.0, 2.0]));
      this._material.setParameter('lightIntensity', 0.8);
      this._material.setParameter('ambientStrength', 0.3);
    }

    // Create mesh instance with material
    const mi = new pc.MeshInstance(mesh, this._material);

    // Set up instancing (skip for debug shader)
    let visCount = 0;
    if (!DEBUG_SHADER) {
      const instFormat = createInstancingFormat(device);
      const segmentArray = packSegmentData(data);
      visCount = segmentArray.length / SEG_FLOATS;
      const instVb = new pc.VertexBuffer(device, instFormat, visCount, {
        data: segmentArray.buffer as ArrayBuffer,
      });
      mi.setInstancing(instVb);
      mi.instancingCount = visCount;

      console.log(`[PlayCanvas] Segment mesh: ${visCount} visible instances (${data.count} total), ${bodyGeo.indices.length} idx`);
      this.stats.triangles = Math.round((bodyGeo.indices.length / 3) * visCount / 1000);
    } else {
      console.log(`[PlayCanvas] Debug mesh: ${ringVertCount} verts, ${bodyGeo.indices.length} idx`);
    }

    // Fit camera to model bounds
    this._fitCamera(data);

    // Segment mesh entity at model center
    const segEntity = new pc.Entity('segments');
    segEntity.addComponent('render', {
      meshInstances: [mi],
    });
    segEntity.setLocalPosition(this._orbitTarget.x, this._orbitTarget.y, this._orbitTarget.z);
    this.app.root.addChild(segEntity);
    this._meshEntity = segEntity;

    const elapsed = performance.now() - t0;
    return Math.round(elapsed);
  }

  setMaterial(props: MaterialProps): void {
    // Wire material properties to shader uniforms
    if (this._material) {
      // For now, just log — full PBR material binding will come when we
      // integrate PlayCanvas's PBR shader chunks
      console.log('[PlayCanvas] setMaterial:', props);
    }
  }

  async setEnvMap(url: string): Promise<void> {
    console.log('[PlayCanvas] setEnvMap:', url, '(deferred — IBL integration pending)');
  }

  resize(): void {
    if (!this._canvas || !this._container) return;
    const w = this._container.clientWidth;
    const h = this._container.clientHeight;
    if (w === 0 || h === 0) return;

    const iw = Math.round(w);
    const ih = Math.round(h);
    if (this._canvas.width !== iw || this._canvas.height !== ih) {
      this.app.graphicsDevice.resizeCanvas(iw, ih);
    }
    const cam = this._cameraEntity?.camera;
    if (cam) {
      cam.aspectRatio = iw / ih;
    }
  }

  dispose(): void {
    this._disposed = true;
    this._resizeObserver?.disconnect();
    if (this._statsTimer) clearInterval(this._statsTimer);
    this._meshEntity?.destroy();
    this.app?.destroy();
  }

  getScreenshotHooks(): ScreenshotHooks | null {
    return null;
  }

  // ── Orbit camera ──

  private _orbitTarget = new pc.Vec3(0, 0, 0);
  private _orbitRadius = 200;
  private _orbitAlpha = Math.PI / 4;
  private _orbitBeta = Math.PI / 4;
  private _orbitDirty = true;
  private _animate = false;

  private _setupOrbitControls(canvas: HTMLCanvasElement) {
    let dragging = false;
    let lastX = 0, lastY = 0;

    canvas.addEventListener('pointerdown', (e) => {
      dragging = true;
      lastX = e.clientX;
      lastY = e.clientY;
      canvas.setPointerCapture(e.pointerId);
    });

    canvas.addEventListener('pointermove', (e) => {
      if (!dragging) return;
      const dx = e.clientX - lastX;
      const dy = e.clientY - lastY;
      lastX = e.clientX;
      lastY = e.clientY;
      this._orbitAlpha -= dx * 0.005;
      this._orbitBeta = Math.max(0.01, Math.min(Math.PI - 0.01, this._orbitBeta + dy * 0.005));
      this._orbitDirty = true;
    });

    canvas.addEventListener('pointerup', () => { dragging = false; });

    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      this._orbitRadius *= 1 + e.deltaY * 0.001;
      this._orbitRadius = Math.max(1, Math.min(10000, this._orbitRadius));
      this._orbitDirty = true;
    }, { passive: false });
  }

  private _updateCamera() {
    if (!this._orbitDirty || !this._cameraEntity) return;
    this._orbitDirty = false;

    const r = this._orbitRadius;
    const a = this._orbitAlpha;
    const b = this._orbitBeta;
    const t = this._orbitTarget;

    // Y‑up spherical: beta=π/2 = top, beta=0 = horizon
    this._cameraEntity.setLocalPosition(
      t.x + r * Math.cos(b) * Math.sin(a),
      t.y + r * Math.sin(b),
      t.z + r * Math.cos(b) * Math.cos(a),
    );
    this._cameraEntity.lookAt(t);
  }

  private _fitCamera(data: SegbinData) {
    const g = data.geoms;
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (let i = 0; i < data.count; i++) {
      if (data.roles[i] === 10 || data.roles[i] === 13) continue; // SkirtBrim, Other
      // Rotate Z‑up → Y‑up: (x,y,z) → (x,z,-y)
      const sx = g[i * 8 + 0];      // x
      const sy = g[i * 8 + 2];      // z → y
      const sz = -g[i * 8 + 1];     // -y → z
      const ex = g[i * 8 + 3];
      const ey = g[i * 8 + 5];      // z → y
      const ez = -g[i * 8 + 4];     // -y → z
      if (sx < minX) minX = sx; if (sx > maxX) maxX = sx;
      if (sy < minY) minY = sy; if (sy > maxY) maxY = sy;
      if (sz < minZ) minZ = sz; if (sz > maxZ) maxZ = sz;
      if (ex < minX) minX = ex; if (ex > maxX) maxX = ex;
      if (ey < minY) minY = ey; if (ey > maxY) maxY = ey;
      if (ez < minZ) minZ = ez; if (ez > maxZ) maxZ = ez;
    }
    this._orbitTarget.set((minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2);
    const dx = maxX - minX, dy = maxY - minY, dz = maxZ - minZ;
    this._orbitRadius = Math.sqrt(dx * dx + dy * dy + dz * dz) * 1.5;
    this._orbitDirty = true;
  }

  private _startLoop() {
    if (this._animate) return;
    this._animate = true;
    requestAnimationFrame(this._loop);
  }

  private _loop = () => {
    if (this._disposed || !this._animate) {
      this._animate = false;
      return;
    }
    this._updateCamera();
    this.app.render();
    this._fpsFrames++;

    // Debug: log camera position on first 5 frames
    if (this._fpsFrames <= 5 && this._cameraEntity) {
      const pos = this._cameraEntity.getPosition();
      console.log(`[PlayCanvas] frame ${this._fpsFrames}: camera pos=(${pos.x.toFixed(1)}, ${pos.y.toFixed(1)}, ${pos.z.toFixed(1)}), target=(${this._orbitTarget.x.toFixed(1)}, ${this._orbitTarget.y.toFixed(1)}, ${this._orbitTarget.z.toFixed(1)})`);
    }

    requestAnimationFrame(this._loop);
  };
}
