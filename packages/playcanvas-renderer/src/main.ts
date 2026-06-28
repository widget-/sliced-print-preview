import type { MaterialProps, Renderer, ScreenshotHooks } from '@sliced/shared';
import { loadSegbin, type SegbinData } from '@sliced/shared';
import { generateBodyGeometry, generateCapGeometry } from './geometry';
import * as pc from 'playcanvas';

// ── Segment data per-instance layout ──
// ATTR12: startPos.xyz + width          (vec4)
// ATTR13: endPos.xyz + arcWeight        (vec4) — arcWeight = conic weight (0 for linear)
// ATTR14: color.rgb + flags             (vec4) — flags.x = isArc (>0.5)
// ATTR15: nextStartPos.xyz + _unused    (vec4) — next segment's start (for Bezier)
const SEG_FLOATS = 16;

// Cap instance uses same layout but ATTR14.w = isEnd (>0.5 for end cap)
const CAP_FLOATS = 16;

function packSegmentData(data: SegbinData): Float32Array {
  const n = data.count;
  let visCount = 0;
  for (let i = 0; i < n; i++) {
    const r = data.roles[i];
    if (r !== 10 && r !== 13) visCount++;
  }

  // Build index map: original index → packed index (or -1 if hidden)
  const idxMap = new Int32Array(n).fill(-1);
  {
    let vi = 0;
    for (let i = 0; i < n; i++) {
      if (data.roles[i] !== 10 && data.roles[i] !== 13) idxMap[i] = vi++;
    }
  }

  const out = new Float32Array(visCount * SEG_FLOATS);
  const g = data.geoms;
  for (let i = 0; i < n; i++) {
    const vi = idxMap[i];
    if (vi < 0) continue;
    const r = data.roles[i];
    const off = vi * SEG_FLOATS;

    // ATTR12: startPos.xyz + width  (rotate Z‑up → Y‑up)
    out[off + 0] = g[i * 8 + 0];      // x → x
    out[off + 1] = g[i * 8 + 2];      // z → y
    out[off + 2] = -g[i * 8 + 1];     // -y → z
    out[off + 3] = g[i * 8 + 6];      // width

    // ATTR13: endPos.xyz + arcWeight
    out[off + 4] = g[i * 8 + 3];      // x → x
    out[off + 5] = g[i * 8 + 5];      // z → y
    out[off + 6] = -g[i * 8 + 4];     // -y → z

    // Arc weight: unpack from g[i*8+7] (packed layerZ + conic weight)
    const isArc = (data.segType[i] & 1) !== 0;
    let arcWeight = 0.0;
    if (isArc) {
      const packed = g[i * 8 + 7];
      const lz = Math.round(packed * 100) / 100;
      arcWeight = (packed - lz) * 10000;
    }
    out[off + 7] = arcWeight;

    // ATTR14: color.rgb + flags (isArc bit)
    const c = roleColorVec3(r);
    out[off + 8] = c[0];
    out[off + 9] = c[1];
    out[off + 10] = c[2];
    out[off + 11] = isArc ? 1.0 : 0.0;

    // ATTR15: next segment's startPos (for Bezier evaluation)
    // Find next visible segment
    let nextIdx = i + 1;
    while (nextIdx < n && idxMap[nextIdx] < 0) nextIdx++;
    if (nextIdx < n) {
      out[off + 12] = g[nextIdx * 8 + 0];      // x
      out[off + 13] = g[nextIdx * 8 + 2];      // z → y
      out[off + 14] = -g[nextIdx * 8 + 1];     // -y → z
    } else {
      // Fallback: use endPos
      out[off + 12] = g[i * 8 + 3];
      out[off + 13] = g[i * 8 + 5];
      out[off + 14] = -g[i * 8 + 4];
    }
    out[off + 15] = 0.0;
  }
  return out;
}

function packCapData(data: SegbinData, visSegments: Float32Array): Float32Array {
  // 2 caps per segment (start + end), each with full segment data + isEnd flag
  const segCount = visSegments.length / SEG_FLOATS;
  const capCount = segCount * 2;
  const out = new Float32Array(capCount * CAP_FLOATS);

  for (let ci = 0; ci < capCount; ci++) {
    const segIdx = Math.floor(ci / 2);
    const isEnd = (ci % 2 === 1) ? 1.0 : 0.0;
    const srcOff = segIdx * SEG_FLOATS;
    const dstOff = ci * CAP_FLOATS;

    // Copy full segment data (ATTR12-13, ATTR15)
    for (let j = 0; j < 8; j++) out[dstOff + j] = visSegments[srcOff + j];     // ATTR12 + ATTR13
    for (let j = 12; j < 16; j++) out[dstOff + j] = visSegments[srcOff + j];    // ATTR15

    // ATTR14: color.rgb + isEnd
    out[dstOff + 8] = visSegments[srcOff + 8];  // r
    out[dstOff + 9] = visSegments[srcOff + 9];  // g
    out[dstOff + 10] = visSegments[srcOff + 10]; // b
    out[dstOff + 11] = isEnd;
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
    { semantic: pc.SEMANTIC_ATTR15, components: 4, type: pc.TYPE_FLOAT32 },
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
attribute vec4 instance_line4;
#endif

uniform mat4 matrix_viewProjection;
uniform mat4 matrix_view;

varying vec3 vWorldPos;
varying vec3 vWorldNormal;
varying vec3 vColor;
varying vec4 vClipPos;

const float hScale = 1.25;
const float areaCorrection = 1.1;
const vec3 upDirConst = vec3(0.0, 1.0, 0.0);

mat4 getModelMatrix() { return mat4(1.0); }

void main() {
#ifdef INSTANCING
  vec3 startPos = instance_line1.xyz;
  float width = instance_line1.w;
  vec3 endPos = instance_line2.xyz;
  float arcWeight = instance_line2.w;
  vec3 segColor = instance_line3.rgb;
  float isArcF = instance_line3.a;
  bool isArc = isArcF > 0.5;
  vec3 nextStartPos = instance_line4.xyz;
#else
  vec3 startPos = vec3(-0.5, 0.0, 0.0);
  float width = 0.4;
  vec3 endPos = vec3(0.5, 0.0, 0.0);
  float arcWeight = 0.0;
  vec3 segColor = vec3(1.0, 0.5, 0.2);
  bool isArc = false;
  vec3 nextStartPos = vec3(0.5, 0.0, 0.0);
#endif

  float t = vertex_position.z + 0.5;

  vec3 segPos;
  vec3 endTangent;

  if (isArc && arcWeight > 0.001) {
    // Rational quadratic Bezier
    vec3 p0 = startPos;
    vec3 p1 = endPos;
    vec3 p2 = nextStartPos;
    float w = arcWeight;
    float mt = 1.0 - t;
    float mt2 = mt * mt;
    float t2 = t * t;
    float denom = mt2 + 2.0 * t * mt * w + t2;
    segPos = (mt2 * p0 + 2.0 * t * mt * w * p1 + t2 * p2) / denom;

    // Finite-difference tangent
    float te = min(t + 0.01, 1.0); float me = 1.0 - te;
    float me2 = me * me; float te2 = te * te;
    float de = me2 + 2.0 * te * me * w + te2;
    vec3 pe = (me2 * p0 + 2.0 * te * me * w * p1 + te2 * p2) / de;
    float ts = max(t - 0.01, 0.0); float ms = 1.0 - ts;
    float ms2 = ms * ms; float ts2 = ts * ts;
    float ds = ms2 + 2.0 * ts * ms * w + ts2;
    vec3 ps = (ms2 * p0 + 2.0 * ts * ms * w * p1 + ts2 * p2) / ds;
    vec3 dDir = pe - ps;
    endTangent = length(dDir) < 0.0001 ? vec3(0.0, 1.0, 0.0) : normalize(dDir);
  } else {
    segPos = mix(startPos, endPos, t);
    vec3 dir = endPos - startPos;
    float len = length(dir);
    endTangent = len > 0.001 ? dir / len : vec3(0.0, 1.0, 0.0);
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

// ── GLSL fragment shader — Cook-Torrance GGX with material params ──
const BODY_FRAG_GLSL = `
varying vec3 vWorldPos;
varying vec3 vWorldNormal;
varying vec3 vColor;
varying vec4 vClipPos;

uniform vec3 camera_position;
uniform vec3 lightDir;
uniform float lightIntensity;
uniform float ambientStrength;
uniform float uRoughness;
uniform float uMetalness;
uniform float uEnvIntensity;
uniform float uSpecularStrength;
uniform vec3 uBaseColorTint;

const float PI = 3.14159265;

float ggxDistribution(vec3 N, vec3 H, float a) {
  float a2 = a * a;
  float NdotH = max(dot(N, H), 0.0);
  float NdotH2 = NdotH * NdotH;
  float denom = NdotH2 * (a2 - 1.0) + 1.0;
  return a2 / (PI * denom * denom);
}

float schlickFresnel(float NdotV, float f0) {
  return f0 + (1.0 - f0) * pow(1.0 - NdotV, 5.0);
}

float smithG(vec3 N, vec3 V, vec3 L, float a) {
  float k = (a + 1.0) * (a + 1.0) / 8.0;
  float NdotV = max(dot(N, V), 0.0);
  float NdotL = max(dot(N, L), 0.0);
  return (NdotV / (NdotV * (1.0 - k) + k)) * (NdotL / (NdotL * (1.0 - k) + k));
}

void main() {
  vec3 N = normalize(vWorldNormal);
  vec3 V = normalize(camera_position - vWorldPos);
  vec3 L = normalize(lightDir);
  vec3 H = normalize(L + V);

  float NdotL = max(dot(N, L), 0.0);
  float NdotV = max(dot(N, V), 0.001);

  float roughness = max(uRoughness, 0.05);
  float metalness = clamp(uMetalness, 0.0, 1.0);

  // Base colour: segment role colour tinted by user setting
  vec3 baseColor = vColor * uBaseColorTint;

  // Fresnel f0: blend between dielectric (0.04) and metallic (baseColor)
  vec3 f0 = mix(vec3(0.04), baseColor, metalness);

  float D = ggxDistribution(N, H, roughness * roughness);
  vec3 F = vec3(schlickFresnel(NdotV, dot(f0, vec3(0.333))));
  float G = smithG(N, V, L, roughness * roughness);

  vec3 specular = (D * F * G) / max(4.0 * NdotV * NdotL, 0.001);
  vec3 diffuse = baseColor / PI * (1.0 - mix(0.0, 1.0, metalness));

  vec3 ambient = baseColor * ambientStrength * uEnvIntensity;
  vec3 lit = ambient + (diffuse + specular * uSpecularStrength) * NdotL * lightIntensity * PI;

  gl_FragColor = vec4(lit, 1.0);
}
`;

// ── GLSL cap vertex shader ──
const CAP_VERT_GLSL = `
attribute vec3 vertex_position;
attribute vec3 vertex_normal;

#ifdef INSTANCING
attribute vec4 instance_line1;
attribute vec4 instance_line2;
attribute vec4 instance_line3;
attribute vec4 instance_line4;
#endif

uniform mat4 matrix_viewProjection;

varying vec3 vWorldPos;
varying vec3 vWorldNormal;
varying vec3 vColor;

const float hScale = 1.25;
const float areaCorrection = 1.1;
const vec3 upDirConst = vec3(0.0, 1.0, 0.0);

mat4 getModelMatrix() { return mat4(1.0); }

void main() {
#ifdef INSTANCING
  vec3 startPos = instance_line1.xyz;
  float width = instance_line1.w;
  vec3 endPos = instance_line2.xyz;
  float arcWeight = instance_line2.w;
  vec3 segColor = instance_line3.rgb;
  float isEnd = instance_line3.a;
  vec3 nextStartPos = instance_line4.xyz;
#else
  vec3 startPos = vec3(0.0, 0.0, 0.0);
  float width = 0.4;
  vec3 endPos = vec3(1.0, 0.0, 0.0);
  float arcWeight = 0.0;
  vec3 segColor = vec3(1.0, 0.5, 0.2);
  float isEnd = 1.0;
  vec3 nextStartPos = vec3(1.0, 0.0, 0.0);
#endif

  // Position at start or end of segment
  vec3 pos = isEnd > 0.5 ? endPos : startPos;

  // Tangent direction outward from endpoint
  vec3 dir = endPos - startPos;
  float len = length(dir);
  vec3 tangent = len > 0.001 ? dir / len : vec3(0.0, 1.0, 0.0);

  vec3 upDir = upDirConst;
  vec3 rightDir = -normalize(cross(upDir, tangent));
  if (length(rightDir) < 0.001) rightDir = vec3(1.0, 0.0, 0.0);
  vec3 fwdDir = -normalize(cross(rightDir, upDir));
  mat3 rot = mat3(rightDir, upDir, fwdDir);

  // Flip X and Z for start cap so bulge faces outward
  float flipEnd = isEnd > 0.5 ? 1.0 : -1.0;

  vec3 local = vec3(
    flipEnd * vertex_position.x * width * areaCorrection,
            vertex_position.y * width * hScale,
    flipEnd * vertex_position.z * width * 0.5
  );

  vec3 worldPos = pos + rot * local;

  // Flip local normal Z for start cap
  vec3 localNormal = vec3(
    flipEnd * vertex_normal.x,
    vertex_normal.y,
    flipEnd * vertex_normal.z
  );
  vec3 worldNormal = normalize(rot * localNormal);

  gl_Position = matrix_viewProjection * vec4(worldPos, 1.0);
  vWorldPos = worldPos;
  vWorldNormal = worldNormal;
  vColor = segColor;
}
`;

const CAP_FRAG_GLSL = `
varying vec3 vWorldPos;
varying vec3 vWorldNormal;
varying vec3 vColor;

uniform vec3 lightDir;
uniform float lightIntensity;
uniform float ambientStrength;
uniform float uRoughness;
uniform float uMetalness;
uniform float uEnvIntensity;
uniform float uSpecularStrength;
uniform vec3 uBaseColorTint;

const float PI = 3.14159265;

float ggxDistribution(vec3 N, vec3 H, float a) {
  float a2 = a * a;
  float NdotH = max(dot(N, H), 0.0);
  float NdotH2 = NdotH * NdotH;
  float denom = NdotH2 * (a2 - 1.0) + 1.0;
  return a2 / (PI * denom * denom);
}
float schlickFresnel(float NdotV, float f0) {
  return f0 + (1.0 - f0) * pow(1.0 - NdotV, 5.0);
}
float smithG(vec3 N, vec3 V, vec3 L, float a) {
  float k = (a + 1.0) * (a + 1.0) / 8.0;
  float NdotV = max(dot(N, V), 0.0);
  float NdotL = max(dot(N, L), 0.0);
  return (NdotV / (NdotV * (1.0 - k) + k)) * (NdotL / (NdotL * (1.0 - k) + k));
}

void main() {
  vec3 N = normalize(vWorldNormal);
  // Caps don't have camera position — estimate view dir
  vec3 V = normalize(vec3(0.0, 0.0, 1.0));
  vec3 L = normalize(lightDir);
  vec3 H = normalize(L + V);

  float NdotL = max(dot(N, L), 0.0);
  float NdotV = max(dot(N, V), 0.001);

  float roughness = max(uRoughness, 0.05);
  float metalness = clamp(uMetalness, 0.0, 1.0);
  vec3 baseColor = vColor * uBaseColorTint;
  vec3 f0 = mix(vec3(0.04), baseColor, metalness);

  float D = ggxDistribution(N, H, roughness * roughness);
  vec3 F = vec3(schlickFresnel(NdotV, dot(f0, vec3(0.333))));
  float G = smithG(N, V, L, roughness * roughness);

  vec3 specular = (D * F * G) / max(4.0 * NdotV * NdotL, 0.001);
  vec3 diffuse = baseColor / PI * (1.0 - mix(0.0, 1.0, metalness));
  vec3 ambient = baseColor * ambientStrength * uEnvIntensity;
  vec3 lit = ambient + (diffuse + specular * uSpecularStrength) * NdotL * lightIntensity * PI;

  gl_FragColor = vec4(lit, 1.0);
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
  private _cameraFrame?: pc.CameraFrame;
  private _material?: pc.ShaderMaterial;
  private _capMaterial?: pc.ShaderMaterial;

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

    // Start app (initializes component systems, required for CameraFrame)
    app.start();

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

    // ── CameraFrame (SSAO + TAA + tone mapping) ──
    const cameraFrame = new pc.CameraFrame(app, this._cameraEntity.camera!);
    this._cameraFrame = cameraFrame;
    cameraFrame.ssao.type = pc.SSAOTYPE_COMBINE;
    cameraFrame.ssao.blurEnabled = true;
    cameraFrame.taa.enabled = true;
    cameraFrame.taa.jitter = 1.0;
    cameraFrame.bloom.enabled = true;
    cameraFrame.bloom.intensity = 0.3;
    // Keep default tone mapping (ACES)

    console.log('[PlayCanvas] CameraFrame enabled: SSAO + TAA + Bloom');

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

    // Reverse triangle winding: swap second/third vertex in each triangle
    // (Y‑up basis has opposite handedness from the geometry's Z‑up origin)
    const revIndices = new Uint16Array(bodyGeo.indices.length);
    for (let i = 0; i < bodyGeo.indices.length; i += 3) {
      revIndices[i + 0] = bodyGeo.indices[i + 0];
      revIndices[i + 1] = bodyGeo.indices[i + 2]; // swap
      revIndices[i + 2] = bodyGeo.indices[i + 1];
    }
    const ib = new pc.IndexBuffer(device, pc.INDEXFORMAT_UINT16, revIndices.length, pc.BUFFER_STATIC, revIndices.buffer as ArrayBuffer);

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
        instance_line4: pc.SEMANTIC_ATTR15,
      },
    };
    this._material = new pc.ShaderMaterial(shaderDesc);

    // Disable instancing temporarily for debug shader
    // (instancing attributes not declared in debug shader)

    // Back-face culling (winding reversed above to match Y‑up)
    this._material.cull = pc.CULLFACE_BACK;

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
    let segmentArray: Float32Array | null = null;
    if (!DEBUG_SHADER) {
      const instFormat = createInstancingFormat(device);
      segmentArray = packSegmentData(data);
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

    // ── Caps ──
    if (!DEBUG_SHADER && segmentArray) {
      const capGeo = generateCapGeometry(0.35, 0.35, 5, 4);
      const capVertCount = capGeo.interleaved.length / 6;

      // Create cap mesh
      const capMesh = new pc.Mesh(device);
      const capInterleaved = new Float32Array(capGeo.interleaved);
      const capVb = new pc.VertexBuffer(device, interleavedFormat, capVertCount, {
        data: capInterleaved.buffer as ArrayBuffer,
      });
      const capIb = new pc.IndexBuffer(device, pc.INDEXFORMAT_UINT16, capGeo.indices.length, pc.BUFFER_STATIC, capGeo.indices.buffer as ArrayBuffer);
      capMesh.vertexBuffer = capVb;
      capMesh.indexBuffer = [capIb];
      capMesh.primitive = [{
        type: pc.PRIMITIVE_TRIANGLES,
        base: 0,
        count: capGeo.indices.length,
        indexed: true,
      }];
      capMesh.update();

      // Cap shader material
      const capShaderDesc = {
        uniqueName: 'segment-cap',
        vertexGLSL: CAP_VERT_GLSL,
        fragmentGLSL: CAP_FRAG_GLSL,
        attributes: {
          vertex_position: pc.SEMANTIC_POSITION,
          vertex_normal: pc.SEMANTIC_NORMAL,
          instance_line1: pc.SEMANTIC_ATTR12,
          instance_line2: pc.SEMANTIC_ATTR13,
          instance_line3: pc.SEMANTIC_ATTR14,
          instance_line4: pc.SEMANTIC_ATTR15,
        },
      };
      const capMaterial = new pc.ShaderMaterial(capShaderDesc);
      this._capMaterial = capMaterial;
      capMaterial.defines.set('INSTANCING', '');
      capMaterial.cull = pc.CULLFACE_NONE; // caps need reverse winding per WebGPU
      capMaterial.setParameter('lightDir', new Float32Array([1.0, 1.0, 2.0]));
      capMaterial.setParameter('lightIntensity', 0.8);
      capMaterial.setParameter('ambientStrength', 0.3);

      // Cap instancing data
      const capData = packCapData(data, segmentArray);
      const capCount = capData.length / CAP_FLOATS;
      const capInstVb = new pc.VertexBuffer(device, createInstancingFormat(device), capCount, {
        data: capData.buffer as ArrayBuffer,
      });

      const capMi = new pc.MeshInstance(capMesh, capMaterial);
      capMi.setInstancing(capInstVb);
      capMi.instancingCount = capCount;

      const capEntity = new pc.Entity('caps');
      capEntity.addComponent('render', {
        meshInstances: [capMi],
      });
      capEntity.setLocalPosition(this._orbitTarget.x, this._orbitTarget.y, this._orbitTarget.z);
      this.app.root.addChild(capEntity);

      console.log(`[PlayCanvas] Caps: ${capCount} instances, ${capGeo.indices.length} idx`);
      this.stats.triangles += Math.round((capGeo.indices.length / 3) * capCount / 1000);

      // Apply current material properties to caps
      this._writeMaterialParams();
    }

    const elapsed = performance.now() - t0;
    return Math.round(elapsed);
  }

  setMaterial(props: MaterialProps): void {
    this._materialProps = props;
    this._writeMaterialParams();
  }

  private _materialProps?: MaterialProps;

  private _writeMaterialParams() {
    const p = this._materialProps;
    if (!p) return;

    const tint: [number, number, number] = [
      parseInt(p.baseColorTint.slice(1, 3), 16) / 255,
      parseInt(p.baseColorTint.slice(3, 5), 16) / 255,
      parseInt(p.baseColorTint.slice(5, 7), 16) / 255,
    ];

    const mats = [this._material, this._capMaterial];
    for (const mat of mats) {
      if (!mat) continue;
      mat.setParameter('uRoughness', p.roughness);
      mat.setParameter('uMetalness', p.metalness);
      mat.setParameter('uEnvIntensity', p.envIntensity);
      mat.setParameter('uSpecularStrength', p.specularStrength);
      mat.setParameter('ambientStrength', p.ambientStrength);
      mat.setParameter('uBaseColorTint', new Float32Array(tint));
    }
  }

  setSSAOIntensity(v: number) {
    if (this._cameraFrame) this._cameraFrame.ssao.intensity = v;
  }

  setSSAORadius(v: number) {
    if (this._cameraFrame) this._cameraFrame.ssao.radius = v;
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

  private _lastFrameMs = 0;

  private _loop = () => {
    if (this._disposed || !this._animate) {
      this._animate = false;
      return;
    }

    const now = performance.now();
    const dt = this._lastFrameMs ? (now - this._lastFrameMs) / 1000 : 0.016;
    this._lastFrameMs = now;

    this._updateCamera();
    this.app.update(dt);
    this.app.render();
    this._fpsFrames++;
    this._fpsFrames++;

    // Debug: log camera position on first 5 frames
    if (this._fpsFrames <= 5 && this._cameraEntity) {
      const pos = this._cameraEntity.getPosition();
      console.log(`[PlayCanvas] frame ${this._fpsFrames}: camera pos=(${pos.x.toFixed(1)}, ${pos.y.toFixed(1)}, ${pos.z.toFixed(1)}), target=(${this._orbitTarget.x.toFixed(1)}, ${this._orbitTarget.y.toFixed(1)}, ${this._orbitTarget.z.toFixed(1)})`);
    }

    requestAnimationFrame(this._loop);
  };
}
