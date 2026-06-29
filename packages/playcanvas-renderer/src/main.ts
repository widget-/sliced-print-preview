import type { MaterialProps, Renderer, ScreenshotHooks } from '@sliced/shared';
import { loadSegbin, type SegbinData } from '@sliced/shared';
import { generateBodyGeometry, generateCapGeometry } from './geometry';
import * as pc from 'playcanvas';
import ssaoEngineGLSL from './ssao-engine.glsl.js';

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

  // Wrap diffuse: prevents back faces from going completely black
  // when ambient is low. Maps NdotL from [-1,1] to [wrap/(1+wrap), 1].
  float wrap = 0.25;
  float NdotL_wrapped = (NdotL + wrap) / (1.0 + wrap);

  vec3 ambient = baseColor * ambientStrength * uEnvIntensity;
  vec3 lit = ambient + (diffuse + specular * uSpecularStrength) * NdotL_wrapped * lightIntensity * PI;

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
uniform vec3 camera_position;
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
  vec3 baseColor = vColor * uBaseColorTint;
  vec3 f0 = mix(vec3(0.04), baseColor, metalness);

  float D = ggxDistribution(N, H, roughness * roughness);
  vec3 F = vec3(schlickFresnel(NdotV, dot(f0, vec3(0.333))));
  float G = smithG(N, V, L, roughness * roughness);

  vec3 specular = (D * F * G) / max(4.0 * NdotV * NdotL, 0.001);
  vec3 diffuse = baseColor / PI * (1.0 - mix(0.0, 1.0, metalness));
  // Wrap diffuse: prevents back faces from going completely black
  // when ambient is low. Maps NdotL from [-1,1] to [wrap/(1+wrap), 1].
  float wrap = 0.25;
  float NdotL_wrapped = (NdotL + wrap) / (1.0 + wrap);

  vec3 ambient = baseColor * ambientStrength * uEnvIntensity;
  vec3 lit = ambient + (diffuse + specular * uSpecularStrength) * NdotL_wrapped * lightIntensity * PI;

  gl_FragColor = vec4(lit, 1.0);
}
`;

// ── Debug depth visualization shader (fullscreen quad) ──
// Log-scale mapping: near=bright, far=dark, with visible gradation
const DEBUG_DEPTH_FRAG_GLSL = `
varying vec2 uv0;
uniform sampler2D uSceneDepthMap;
void main() {
  float linearDepth = texture2D(uSceneDepthMap, uv0).r;
  // Reciprocal mapping: depth 0->1.0, depth 200->0.5, depth 1000->0.17
  float v = 1.0 / (1.0 + linearDepth * 0.005);
  gl_FragColor = vec4(v, v, v, 1.0);
}
`;

// ── Depth-only fragment shader (writes linear view-space Z to R channel) ──
const DEPTH_ONLY_FRAG_GLSL = `
varying vec3 vWorldPos;
uniform mat4 matrix_view;
void main() {
  float viewZ = -(matrix_view * vec4(vWorldPos, 1.0)).z;
  gl_FragColor = vec4(viewZ, 0.0, 0.0, 1.0);
}
`;

const SSAO_COMPOSITE_FRAG_GLSL = `
varying vec2 uv0;
uniform sampler2D uSceneTexture;
uniform sampler2D uAOTexture;
void main() {
  vec4 scene = texture2D(uSceneTexture, uv0);
  float ao = texture2D(uAOTexture, uv0).r;
  gl_FragColor = vec4(scene.rgb * ao, scene.a);
}
`;

// ── World-space depth-aware bilateral blur shader ──
// Takes a world-space blur radius and projects it to screen-space per pixel,
// so the blur covers the same world-space area regardless of camera distance.
const SSAO_BLUR_FRAG_GLSL = `
varying vec2 uv0;
uniform sampler2D uDepthMap;
uniform sampler2D sourceTexture;
uniform vec2 sourceInvResolution;
uniform float uWorldBlurRadius;
uniform float uProjectionScale;

float random(const highp vec2 w) {
  const vec3 m = vec3(0.06711056, 0.00583715, 52.9829189);
  return fract(m.z * fract(dot(w, m.xy)));
}

float getLinearDepth(vec2 uv) { return texture2D(uDepthMap, uv).r; }
float getLinearScreenDepth(vec2 uv) { return getLinearDepth(uv); }

mediump float bilateralWeight(in mediump float depth, in mediump float sampleDepth) {
  mediump float diff = (sampleDepth - depth);
  return max(0.0, 1.0 - diff * diff);
}

void tap(inout float sum, inout float totalWeight, float weight, float depth, vec2 position) {
  mediump float color = texture2D(sourceTexture, position).r;
  mediump float textureDepth = -getLinearScreenDepth(position);
  mediump float bilateral = bilateralWeight(depth, textureDepth);
  bilateral *= weight;
  sum += color * bilateral;
  totalWeight += bilateral;
}

void main() {
  mediump float depth = -getLinearScreenDepth(uv0);
  mediump float totalWeight = 1.0;
  mediump float color = texture2D(sourceTexture, uv0).r;
  mediump float sum = color * totalWeight;

  // Project world-space blur radius to screen-space pixels
  mediump float ssBlurPixels = uWorldBlurRadius * uProjectionScale / max(-depth, 0.001);
  int filterSize = int(min(ceil(ssBlurPixels), 20.0));

  for (int i = -filterSize; i <= filterSize; i++) {
    mediump float weight = 1.0;

    #ifdef HORIZONTAL
      vec2 offset = vec2(i, 0) * sourceInvResolution;
    #else
      vec2 offset = vec2(0, i) * sourceInvResolution;
    #endif

    tap(sum, totalWeight, weight, depth, uv0 + offset);
  }

  mediump float ao = sum / totalWeight;
  gl_FragColor = vec4(ao, ao, ao, 1.0);
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
  private _keyLight?: pc.Entity;
  private _fillLight?: pc.Entity;
  private _cameraFrame?: pc.CameraFrame;
  private _material?: pc.ShaderMaterial;
  private _capMaterial?: pc.ShaderMaterial;
  private _debugMode = 'none';
  private _debugShader?: pc.Shader;
  // Standalone SSAO pipeline
  private _ssaoDepthRt?: pc.RenderTarget;
  private _ssaoAORt?: pc.RenderTarget;
  private _sceneColorRt?: pc.RenderTarget;
  private _ssaoDepthMaterial?: pc.ShaderMaterial;
  private _ssaoDepthCapMaterial?: pc.ShaderMaterial;
  private _ssaoComputeShader?: pc.Shader;
  private _ssaoClearShader?: pc.Shader;
  private _ssaoCompositeShader?: pc.Shader;
  private _ssaoBlurTempRt?: pc.RenderTarget;
  private _ssaoBlurShaderH?: pc.Shader;
  private _ssaoBlurShaderV?: pc.Shader;
  private _ssaoDirty = true;
  private _ssaoEnabled = true;
  private _ssaoIntensity = 0.5;
  private _ssaoRadius = 0.5;
  private _ssaoPower = 6.0;

  stats = { fps: 0, triangles: 0 };

  async mount(container: HTMLElement, canvas: HTMLCanvasElement): Promise<void> {
    this._canvas = canvas;
    this._container = container;

    this.app = new pc.Application(canvas, {
      graphicsDeviceOptions: {
        antialias: false,  // TAA handles antialiasing; hw MSAA causes blit format warnings
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
    this._keyLight = keyLight;

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
    this._fillLight = fillLight;

    app.scene.ambientLight = new pc.Color(0.3, 0.3, 0.35);

    // ── CameraFrame (SSAO + TAA + tone mapping) ──
    const cameraFrame = new pc.CameraFrame(app, this._cameraEntity.camera!);
    this._cameraFrame = cameraFrame;
    cameraFrame.rendering.sceneDepthMap = true;   // required for SSAO depth prepass
    cameraFrame.ssao.type = pc.SSAOTYPE_NONE;       // disabled: ShaderMaterial can't output linear depth for prepass
    cameraFrame.taa.enabled = true;
    cameraFrame.taa.jitter = 1.0;
    cameraFrame.update();                          // commit all settings

    console.log('[PlayCanvas] CameraFrame enabled: SSAO (COMBINE) + TAA');

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

  setShadowSoftness(v: number) {
    // Map 0-5 slider to PCF levels: 0-1=PCF1, 1-3=PCF3, 3-5=PCF5
    const type = v <= 1 ? pc.SHADOW_PCF1 : v <= 3 ? pc.SHADOW_PCF3 : pc.SHADOW_PCF5;
    if (this._keyLight?.light) this._keyLight.light.shadowType = type;
    if (this._fillLight?.light) this._fillLight.light.shadowType = type;
  }

  setKeyLightIntensity(v: number) {
    if (this._keyLight?.light) this._keyLight.light.intensity = v;
  }

  setFillLightIntensity(v: number) {
    if (this._fillLight?.light) this._fillLight.light.intensity = v;
  }

  setSSAOIntensity(v: number) {
    this._ssaoIntensity = v;
  }

  setSSAORadius(v: number) {
    this._ssaoRadius = v;
  }

  set ssaoEnabled(v: boolean) {
    this._ssaoEnabled = v;
  }

  set debugPreview(v: string) {
    this._debugMode = v ?? 'none';
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
    this._cameraFrame?.destroy();
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

  // ── Standalone SSAO pipeline ──

  private _setupSsaoPipeline() {
    const device = this.app.graphicsDevice;
    const w = device.width;
    const h = device.height;
    if (w === 0 || h === 0) return;

    // Lazy-create / resize render targets
    if (!this._ssaoDepthRt || this._ssaoDepthRt.width !== w || this._ssaoDepthRt.height !== h) {
      this._ssaoDepthRt?.destroy();
      this._ssaoAORt?.destroy();
      this._sceneColorRt?.destroy();
      this._ssaoBlurTempRt?.destroy();

      const depthTex = new pc.Texture(device, {
        name: 'SSAODepth', width: w, height: h,
        format: pc.PIXELFORMAT_R32F,
        mipmaps: false, minFilter: pc.FILTER_NEAREST, magFilter: pc.FILTER_NEAREST,
        addressU: pc.ADDRESS_CLAMP_TO_EDGE, addressV: pc.ADDRESS_CLAMP_TO_EDGE,
      });
      this._ssaoDepthRt = new pc.RenderTarget({ colorBuffer: depthTex, depth: true });

      const aoTex = new pc.Texture(device, {
        name: 'SSAOAO', width: w, height: h,
        format: pc.PIXELFORMAT_R8,
        mipmaps: false, minFilter: pc.FILTER_LINEAR, magFilter: pc.FILTER_LINEAR,
        addressU: pc.ADDRESS_CLAMP_TO_EDGE, addressV: pc.ADDRESS_CLAMP_TO_EDGE,
      });
      this._ssaoAORt = new pc.RenderTarget({ colorBuffer: aoTex, depth: false });

      const sceneTex = new pc.Texture(device, {
        name: 'SSAOScene', width: w, height: h,
        format: pc.PIXELFORMAT_RGBA8,
        mipmaps: false, minFilter: pc.FILTER_LINEAR, magFilter: pc.FILTER_LINEAR,
        addressU: pc.ADDRESS_CLAMP_TO_EDGE, addressV: pc.ADDRESS_CLAMP_TO_EDGE,
      });
      this._sceneColorRt = new pc.RenderTarget({ colorBuffer: sceneTex, depth: false });

      const blurTex = new pc.Texture(device, {
        name: 'SSAOBlurTemp', width: w, height: h,
        format: pc.PIXELFORMAT_R8,
        mipmaps: false, minFilter: pc.FILTER_LINEAR, magFilter: pc.FILTER_LINEAR,
        addressU: pc.ADDRESS_CLAMP_TO_EDGE, addressV: pc.ADDRESS_CLAMP_TO_EDGE,
      });
      this._ssaoBlurTempRt = new pc.RenderTarget({ colorBuffer: blurTex, depth: false });

      // Lazy-create custom world-space depth-aware blur shaders
      // (replaces engine's RenderPassDepthAwareBlur which uses fixed filterSize=4)
      if (!this._ssaoBlurShaderH) {
        this._ssaoBlurShaderH = pc.ShaderUtils.createShader(device, {
          uniqueName: 'SsaoBlurH',
          attributes: { aPosition: pc.SEMANTIC_POSITION },
          vertexGLSL: `attribute vec2 aPosition; varying vec2 uv0;
void main() { gl_Position = vec4(aPosition, 0.5, 1.0); uv0 = aPosition.xy * 0.5 + 0.5; }`,
          fragmentGLSL: `#define HORIZONTAL\n${SSAO_BLUR_FRAG_GLSL}`,
        });
        this._ssaoBlurShaderV = pc.ShaderUtils.createShader(device, {
          uniqueName: 'SsaoBlurV',
          attributes: { aPosition: pc.SEMANTIC_POSITION },
          vertexGLSL: `attribute vec2 aPosition; varying vec2 uv0;
void main() { gl_Position = vec4(aPosition, 0.5, 1.0); uv0 = aPosition.xy * 0.5 + 0.5; }`,
          fragmentGLSL: SSAO_BLUR_FRAG_GLSL,
        });
      }

      this._ssaoDirty = true;
    }

    // Lazy-create depth-only materials
    if (this._ssaoDirty && this._material) {
      // Body depth material — reuse body vertex shader, pair with depth-only fragment
      if (!this._ssaoDepthMaterial) {
        this._ssaoDepthMaterial = new pc.ShaderMaterial({
          uniqueName: 'segment-body-depth',
          vertexGLSL: BODY_VERT_GLSL,
          fragmentGLSL: DEPTH_ONLY_FRAG_GLSL,
          attributes: {
            vertex_position: pc.SEMANTIC_POSITION,
            vertex_normal: pc.SEMANTIC_NORMAL,
            instance_line1: pc.SEMANTIC_ATTR12,
            instance_line2: pc.SEMANTIC_ATTR13,
            instance_line3: pc.SEMANTIC_ATTR14,
            instance_line4: pc.SEMANTIC_ATTR15,
          },
        });
        this._ssaoDepthMaterial.defines.set('INSTANCING', '');
      }

      // Cap depth material
      if (!this._ssaoDepthCapMaterial) {
        this._ssaoDepthCapMaterial = new pc.ShaderMaterial({
          uniqueName: 'segment-cap-depth',
          vertexGLSL: CAP_VERT_GLSL,
          fragmentGLSL: DEPTH_ONLY_FRAG_GLSL,
          attributes: {
            vertex_position: pc.SEMANTIC_POSITION,
            vertex_normal: pc.SEMANTIC_NORMAL,
            instance_line1: pc.SEMANTIC_ATTR12,
            instance_line2: pc.SEMANTIC_ATTR13,
            instance_line3: pc.SEMANTIC_ATTR14,
            instance_line4: pc.SEMANTIC_ATTR15,
          },
        });
        this._ssaoDepthCapMaterial.defines.set('INSTANCING', '');
      }

      // Lazy-create compute + composite shaders
      if (!this._ssaoComputeShader) {
        this._ssaoComputeShader = pc.ShaderUtils.createShader(device, {
          uniqueName: 'SSAOComputeV2',
          attributes: { aPosition: pc.SEMANTIC_POSITION },
          vertexGLSL: `attribute vec2 aPosition; varying vec2 uv0;
void main() { gl_Position = vec4(aPosition, 0.5, 1.0); uv0 = aPosition.xy * 0.5 + 0.5; }`,
          fragmentGLSL: ssaoEngineGLSL,
        });
      }
      if (!this._ssaoCompositeShader) {
        this._ssaoCompositeShader = pc.ShaderUtils.createShader(device, {
          uniqueName: 'SSAOComposite',
          attributes: { aPosition: pc.SEMANTIC_POSITION },
          vertexGLSL: `attribute vec2 aPosition; varying vec2 uv0;
void main() { gl_Position = vec4(aPosition, 0.5, 1.0); uv0 = aPosition.xy * 0.5 + 0.5; }`,
          fragmentGLSL: SSAO_COMPOSITE_FRAG_GLSL,
        });
      }
      this._ssaoDirty = false;
    }
  }

  private _renderSsao() {
    const device = this.app.graphicsDevice;
    const w = device.width;
    const h = device.height;
    this._setupSsaoPipeline();

    const meshEntity = this._meshEntity;
    if (!meshEntity || !this._ssaoDepthRt || !this._ssaoDepthMaterial) return;

    // ── 1. Depth pass: render meshes to depth RT ──
    // Collect all mesh instances from body + cap entities
    const allInstances: pc.MeshInstance[] = [];
    for (const mi of meshEntity.render?.meshInstances ?? []) {
      allInstances.push(mi);
    }
    const capEntity = this.app.root.findByName('caps');
    for (const mi of capEntity?.render?.meshInstances ?? []) {
      allInstances.push(mi);
    }
    if (allInstances.length === 0) return;

    // Swap materials for depth pass
    const saved: pc.ShaderMaterial[] = allInstances.map(mi => mi.material as pc.ShaderMaterial);
    for (const mi of allInstances) {
      const isCap = mi.material === this._capMaterial || saved[allInstances.indexOf(mi)] === this._capMaterial;
      // Actually, just check if mesh is from cap entity
    }
    // Use a simpler approach: just assign depth materials based on entity source
    const bodyInsts = meshEntity.render?.meshInstances ?? [];
    const capInsts = capEntity?.render?.meshInstances ?? [];
    const savedBody: pc.ShaderMaterial[] = [];
    const savedCap: pc.ShaderMaterial[] = [];
    for (const mi of bodyInsts) { savedBody.push(mi.material as pc.ShaderMaterial); mi.material = this._ssaoDepthMaterial; }
    for (const mi of capInsts) { savedCap.push(mi.material as pc.ShaderMaterial); mi.material = this._ssaoDepthCapMaterial; }

    // ── 1. Pre-fill depth RT with far-clip (avoids lazy-init + camera clearColor hack) ──
    if (!this._ssaoClearShader) {
      this._ssaoClearShader = pc.ShaderUtils.createShader(device, {
        uniqueName: 'SSAOClear',
        attributes: { aPosition: pc.SEMANTIC_POSITION },
        vertexGLSL: `attribute vec2 aPosition; varying vec2 uv0;
void main() { gl_Position = vec4(aPosition, 0.0, 1.0); uv0 = aPosition.xy * 0.5 + 0.5; }`,
        fragmentGLSL: `varying vec2 uv0;
void main() { gl_FragColor = vec4(1000.0, 0.0, 0.0, 1.0); }`,
      });
    }
    pc.drawQuadWithShader(device, this._ssaoDepthRt, this._ssaoClearShader);

    // ── 2. Depth pass: render meshes with depth-only shader ──
    const cameraComponent = this._cameraEntity!.camera!;
    const camera = cameraComponent.camera;
    const renderer = (this.app as any).renderer;
    renderer.renderForwardLayer(camera, this._ssaoDepthRt, null, false, pc.SHADER_FORWARD, null, {
      meshInstances: [...bodyInsts, ...capInsts],
      clearDepth: true,
    });

    // Restore materials
    for (let i = 0; i < bodyInsts.length; i++) bodyInsts[i].material = savedBody[i];
    for (let i = 0; i < capInsts.length; i++) capInsts[i].material = savedCap[i];

    // ── 3. SSAO compute pass (verbatim engine SAO) ──
    const scope = device.scope;
    const sampleCount = 16.0;
    const spiralTurns = 10.0;
    const angleInc = (1.0 / (sampleCount - 0.5)) * spiralTurns * 2.0 * Math.PI;
    const invRadius = 1.0 / Math.max(this._ssaoRadius * this._ssaoRadius, 0.0001);
    const minHorizonSin2 = Math.sin(10.0 * Math.PI / 180.0) ** 2;
    const projectionScale = 0.5 * h;  // engine: 0.5 * sourceTexture.height

    scope.resolve('uDepthMap').setValue(this._ssaoDepthRt.colorBuffer);
    scope.resolve('uInvResolution').setValue([1 / w, 1 / h]);
    scope.resolve('uAspect').setValue(w / h);
    scope.resolve('uInvRadiusSquared').setValue(invRadius);
    scope.resolve('uProjectionScaleRadius').setValue(this._ssaoRadius * projectionScale);
    scope.resolve('uIntensity').setValue(this._ssaoIntensity);
    scope.resolve('uPower').setValue(this._ssaoPower);
    scope.resolve('uBias').setValue(0.001);
    scope.resolve('uSampleCount').setValue([sampleCount, 1.0 / sampleCount]);
    scope.resolve('uSpiralTurns').setValue(spiralTurns);
    scope.resolve('uAngleIncCosSin').setValue([Math.cos(angleInc), Math.sin(angleInc)]);
    scope.resolve('uMinHorizonAngleSineSquared').setValue(minHorizonSin2);
    scope.resolve('uMaxLevel').setValue(0.0);
    scope.resolve('uPeak2').setValue((0.1 * this._ssaoRadius) ** 2);
    scope.resolve('uRandomize').setValue(0.0); // breaks spiral banding
    pc.drawQuadWithShader(device, this._ssaoAORt!, this._ssaoComputeShader!);

    // ── 4. World-space depth-aware bilateral blur ──
    // H pass: blur AO horizontally → temp RT
    const bw = this._ssaoAORt!.colorBuffer.width;
    const bh = this._ssaoAORt!.colorBuffer.height;
    scope.resolve('sourceTexture').setValue(this._ssaoAORt!.colorBuffer);
    scope.resolve('sourceInvResolution').setValue([1 / bw, 1 / bh]);
    scope.resolve('uWorldBlurRadius').setValue(0.01);
    scope.resolve('uProjectionScale').setValue(0.5 * h);
    pc.drawQuadWithShader(device, this._ssaoBlurTempRt!, this._ssaoBlurShaderH!);

    // V pass: blur vertically → AO RT (reads from temp RT)
    scope.resolve('sourceTexture').setValue(this._ssaoBlurTempRt!.colorBuffer);
    pc.drawQuadWithShader(device, this._ssaoAORt!, this._ssaoBlurShaderV!);

    // ── 5. Grab scene color from back buffer (after CameraFrame) ──
    device.copyRenderTarget(null, this._sceneColorRt!, true, false);

    // ── 6. Composite: scene × AO → back buffer ──
    scope.resolve('uSceneTexture').setValue(this._sceneColorRt!.colorBuffer);
    scope.resolve('uAOTexture').setValue(this._ssaoAORt!.colorBuffer);
    pc.drawQuadWithShader(device, null, this._ssaoCompositeShader!);
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

    // Apply CameraFrame parameter changes each frame
    this._cameraFrame?.update();

    // Set camera position uniform for ShaderMaterial (not auto-provided)
    if (this._cameraEntity) {
      const cp = this._cameraEntity.getPosition();
      const camPos = new Float32Array([cp.x, cp.y, cp.z]);
      this._material?.setParameter('camera_position', camPos);
      this._capMaterial?.setParameter('camera_position', camPos);
    }

    this.app.render();
    this._fpsFrames++;
    this._fpsFrames++;

    // Standalone SSAO pipeline (depth pass → SSAO → composite)
    if (this._ssaoEnabled) {
      this._renderSsao();
    }

    // Debug visualization
    if (this._debugMode !== 'none') {
      this._renderDebugView();
    }

    requestAnimationFrame(this._loop);
  };

  private _renderDebugView() {
    const device = this.app.graphicsDevice;

    // Use our SSAO depth texture for 'depth' mode, AO texture for 'occlusion'
    if (this._debugMode === 'depth' && this._ssaoDepthRt) {
      // Lazy-create depth debug shader
      if (!this._debugShader) {
        this._debugShader = pc.ShaderUtils.createShader(device, {
          uniqueName: 'DebugDepth',
          attributes: { aPosition: pc.SEMANTIC_POSITION },
          vertexGLSL: `attribute vec2 aPosition; varying vec2 uv0;
void main() { gl_Position = vec4(aPosition, 0.5, 1.0); uv0 = aPosition.xy * 0.5 + 0.5; }`,
          fragmentGLSL: DEBUG_DEPTH_FRAG_GLSL,
        });
      }
      device.scope.resolve('uSceneDepthMap').setValue(this._ssaoDepthRt.colorBuffer);
      pc.drawQuadWithShader(device, null, this._debugShader);
    } else if (this._debugMode === 'occlusion' && this._ssaoAORt) {
      // Just copy AO texture to screen
      device.copyRenderTarget(this._ssaoAORt, null, true, false);
    }
  }
}
