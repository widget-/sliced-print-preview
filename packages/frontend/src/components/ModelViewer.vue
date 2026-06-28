<template>
  <div ref="container" class="preview-container">
    <canvas ref="canvasEl" class="render-canvas" />
  </div>
  <div class="stats-overlay" v-if="stats.show">FPS {{ stats.fps }}<br>tris {{ stats.triangles }}k</div>
</template>

<script setup lang="ts">
import { ref, onMounted, watch, onBeforeUnmount } from 'vue';
import { Engine } from '@babylonjs/core/Engines/engine';
import { Scene } from '@babylonjs/core/scene';
import { ArcRotateCamera } from '@babylonjs/core/Cameras/arcRotateCamera';
import { HemisphericLight } from '@babylonjs/core/Lights/hemisphericLight';
import { DirectionalLight } from '@babylonjs/core/Lights/directionalLight';
import { Vector2, Vector3, Matrix } from '@babylonjs/core/Maths/math.vector';
import { Color4, Color3 } from '@babylonjs/core/Maths/math.color';
import { ShaderMaterial } from '@babylonjs/core/Materials/shaderMaterial';
import { RawTexture } from '@babylonjs/core/Materials/Textures/rawTexture';
import { Texture } from '@babylonjs/core/Materials/Textures/texture';
import { RenderTargetTexture } from '@babylonjs/core/Materials/Textures/renderTargetTexture';
import { PostProcess } from '@babylonjs/core/PostProcesses/postProcess';
import { Effect } from '@babylonjs/core/Materials/effect';
import { HDRTools } from '@babylonjs/core/Misc/HighDynamicRange/hdr';
import { MeshBuilder } from '@babylonjs/core/Meshes/meshBuilder';

import { parseSegbin, buildShaderMeshes, Role } from '../renderer/SegbinLoader';
import type { SegbinData } from '@sliced/shared';
import type { BuildResult } from '../renderer/SegbinLoader';
import { LOD_BODY_GEO, LOD_CAP_GEO } from '../renderer/geometry';
import {
  TAA_PIXEL_SHADER,
} from '../renderer/shaders';


const container = ref<HTMLDivElement>();
const canvasEl = ref<HTMLCanvasElement>();

const props = withDefaults(defineProps<{
  segbinUrl: string | null;
  rendererType?: 'webgl2' | 'webgpu';
  roughness?: number;
  metalness?: number;
  envIntensity?: number;
  specularStrength?: number;
  ambientStrength?: number;
  baseColorTint?: string;
  ssaoEnabled?: boolean;
  roleColors?: boolean;
  shadowSoftness?: number;
  keyLightIntensity?: number;
  fillLightIntensity?: number;
  contactShadowDist?: number;
  contactShadowStrength?: number;
  ssaoIntensity?: number;
  ssaoRadius?: number;
  /** Debug preview mode for WebGPU renderer. */
  debugPreview?: 'none' | 'depth' | 'occlusion' | 'color' | 'normal' | 'shadow' | 'shadow2' | 'velocity' | 'composite-taa' | 'blur-temp' | 'brdf-lut' | 'prefilter-up' | 'prefilter-fwd' | 'prefilter-down' | 'source-up' | 'source-fwd' | 'source-down';
  envMapUrl?: string;
}>(), {
  rendererType: 'webgl2',
  roughness: 0.65,
  metalness: 0.0,
  envIntensity: 1.0,
  specularStrength: 1.0,
  ambientStrength: 0.5,
  baseColorTint: '#e8e0d4',
  debugPreview: 'none',
  envMapUrl: 'ferndale_studio_07_1k.hdr',
  shadowSoftness: 2.0,
  keyLightIntensity: 1.0,
  fillLightIntensity: 0.4,
  contactShadowDist: 0.05,
  contactShadowStrength: 0.5,
  ssaoIntensity: 3.0,
  ssaoRadius: 0.06,
});
const emit = defineEmits<{ 'model-loaded': [ms: number] }>();

const stats = ref({ show: false, fps: 0, triangles: 0 });
let statsFrames = 0;
let statsTime = 0;

let segbinData: SegbinData | null = null;
let buildResult: BuildResult | null = null;  // holds all meshes + LOD groups
let segbinMeshes: import('@babylonjs/core').Mesh[] = [];
let loadGen = 0;
let lastTriCount = 0;
let screenshotLodLock = -1; // >= 0 means LOD is locked for screenshot
let webgpuRenderer: any = null; // WebGPURenderer instance (set when rendererType='webgpu')
let webgpuStatsTimer: any = null;

function onMaterialChange() {
  if (webgpuRenderer) {
    webgpuRenderer.setMaterial({
      roughness: props.roughness,
      metalness: props.metalness,
      envIntensity: props.envIntensity,
      specularStrength: props.specularStrength,
      ambientStrength: props.ambientStrength,
      baseColorTint: props.baseColorTint,
    });
    webgpuRenderer.ssaoEnabled = props.ssaoEnabled !== false;
    webgpuRenderer.pipeline.material.useRoleColors = props.roleColors !== false ? 1 : 0;
    webgpuRenderer.pipeline.writeMaterialUBO();
    webgpuRenderer.setShadowSoftness?.(props.shadowSoftness ?? 2.0);
    webgpuRenderer.setKeyLightIntensity?.(props.keyLightIntensity ?? 1.0);
    webgpuRenderer.setFillLightIntensity?.(props.fillLightIntensity ?? 0.4);
    if (props.contactShadowDist !== undefined) { webgpuRenderer.pipeline.contactShadowDist = props.contactShadowDist; }
    webgpuRenderer.setContactShadowStrength?.(props.contactShadowStrength ?? 1.0);
    webgpuRenderer.setSSAOIntensity?.(props.ssaoIntensity ?? 0.35);
    webgpuRenderer.setSSAORadius?.(props.ssaoRadius ?? 0.06);
    return;
  }
  for (const m of segbinMeshes) {
    const mat = m.material as ShaderMaterial;
    mat.setFloat('uRoughness', props.roughness);
    mat.setFloat('uMetalness', props.metalness);
    mat.setFloat('uEnvIntensity', props.envIntensity);
    mat.setFloat('uSpecularStrength', props.specularStrength);
    mat.setFloat('uAmbientStrength', props.ambientStrength);
    mat.setColor3('uBaseColorTint', Color3.FromHexString(props.baseColorTint));
    mat.setFloat('uShadowSoftness', props.shadowSoftness ?? 2.0);
    mat.setFloat('uUseRoleColors', props.roleColors !== false ? 1.0 : 0.0);
    mat.setFloat('uKeyLightIntensity', props.keyLightIntensity ?? 1.0);
    mat.setFloat('uFillLightIntensity', props.fillLightIntensity ?? 0.4);
  }
}

// Auto-apply material props when they change from the parent
watch(() => [props.roughness, props.metalness, props.envIntensity, props.specularStrength, props.ambientStrength, props.baseColorTint, props.ssaoEnabled, props.roleColors, props.shadowSoftness, props.keyLightIntensity, props.fillLightIntensity, props.contactShadowDist, props.contactShadowStrength, props.ssaoIntensity, props.ssaoRadius], onMaterialChange);
// Forward debugPreview to the WebGPU renderer
watch(() => props.debugPreview, (v) => { if (webgpuRenderer) webgpuRenderer.debugPreview = v ?? 'none'; });
// Reload env map when the user picks a different HDRI
watch(() => props.envMapUrl, (url) => {
  if (webgpuRenderer && typeof webgpuRenderer.setEnvMap === 'function') {
    webgpuRenderer.setEnvMap(url);
  } else if (!webgpuRenderer) {
    loadEnvMap(url);
  }
});

let engine: Engine;
let scene: Scene;
let camera: ArcRotateCamera;
let disposed = false;
let sunLight: DirectionalLight;

function log(msg: string, ...args: any[]) {
  console.log(`[ModelViewer] ${msg}`, ...args);
}

let envMapTexture: RawTexture | null = null;
let taaPipeline: any = null; // replaced by custom TAA below

// Shadow render targets
let shadowRT: RenderTargetTexture | null = null;
let shadowRT2: RenderTargetTexture | null = null;
let fillLight: DirectionalLight | null = null;

// Ground plane
let groundMesh: import('@babylonjs/core').Mesh | null = null;
let groundMat: ShaderMaterial | null = null;

// Precomputed model bounding box (for shadow VP)
let modelCenter = new Vector3(0, 0, 0);
let modelExtent = 1;

// ── Custom TAA state ──
let taaPost: PostProcess | null = null;
let taaHistoryTex: Texture | null = null;
let taaFrame = 0;
let depthRenderer: any = null; // Babylon's DepthRenderer

async function loadEnvMap(url?: string) {
  const hdrUrl = url || props.envMapUrl;
  try {
    const resp = await fetch('/' + hdrUrl);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const buf = await resp.arrayBuffer();
    const uint8 = new Uint8Array(buf);

    const hdrInfo = HDRTools.RGBE_ReadHeader(uint8);
    const rgbPixels = HDRTools.RGBE_ReadPixels(uint8, hdrInfo);

    // RGBE_ReadPixels returns RGB (3 floats/pixel). Pad to RGBA.
    const pixelCount = hdrInfo.width * hdrInfo.height;
    const rgbaPixels = new Float32Array(pixelCount * 4);
    for (let i = 0; i < pixelCount; i++) {
      rgbaPixels[i * 4] = rgbPixels[i * 3];
      rgbaPixels[i * 4 + 1] = rgbPixels[i * 3 + 1];
      rgbaPixels[i * 4 + 2] = rgbPixels[i * 3 + 2];
      rgbaPixels[i * 4 + 3] = 1.0;
    }

    if (envMapTexture) envMapTexture.dispose();

    envMapTexture = new RawTexture(
      rgbaPixels,
      hdrInfo.width,
      hdrInfo.height,
      Engine.TEXTUREFORMAT_RGBA,
      scene!,
      true,   // generateMipMaps
      false,  // invertY
      Texture.TRILINEAR_SAMPLINGMODE,
      Engine.TEXTURETYPE_FLOAT,
    );

    // Apply to all segment meshes
    for (const m of segbinMeshes) {
      const mat = m.material as ShaderMaterial;
      mat.setTexture('uEnvMapEQ', envMapTexture);
      mat.setFloat('uEnvMapLOD', 8.0);
      mat.setFloat('uEnvIntensity', 1.0);
    }

    log('Env map loaded');
  } catch (_err) {
    log('Env map load failed — using hemisphere fallback');
  }
}

function resize() {
  if (webgpuRenderer) { webgpuRenderer.resize(); return; }
  if (!container.value || disposed) return;
  if (!engine) return; // not initialized yet
  const { clientWidth, clientHeight } = container.value;
  if (clientWidth === 0 || clientHeight === 0) return;
  engine.resize();
}

let resizeObserver: ResizeObserver | null = null;

onMounted(() => {
  if (container.value) {
    resizeObserver = new ResizeObserver(() => resize());
    resizeObserver.observe(container.value);
  }
});

async function loadSegbinModel(url: string) {
  log(`Loading segbin: ${url}`);
  const t0 = performance.now();
  const gen = ++loadGen;

  // Remove old meshes
  for (const m of segbinMeshes) {
    m.dispose();
  }
  segbinMeshes = [];

  const resp = await fetch(url);
  if (gen !== loadGen) return;
  const buf = await resp.arrayBuffer();
  if (gen !== loadGen) return;
  const fetchMs = performance.now() - t0;
  log(`  fetch: ${(fetchMs).toFixed(0)}ms (${(buf.byteLength / 1024 / 1024).toFixed(1)} MB)`);

  const tParse = performance.now();
  segbinData = parseSegbin(buf);
  const parseMs = performance.now() - tParse;
  log(`  parse: ${(parseMs).toFixed(0)}ms → ${segbinData.count} segments`);

  const HIDDEN_ROLES: Set<number> = new Set([Role.SkirtBrim, Role.Other]);

  const tBuild = performance.now();
  const result = buildShaderMeshes(segbinData, scene, HIDDEN_ROLES);
  segbinMeshes = result.meshes;
  buildResult = result;
  log(`  Meshes created: ${result.meshes.length} (3 LOD levels), ${segbinData.count} instances`);
  const buildMs = performance.now() - tBuild;
  log(`  build: ${(buildMs).toFixed(0)}ms → ${result.meshes.length} meshes`);

  // Create shadow render targets
  if (!shadowRT) {
    shadowRT = createShadowRT('shadowRT', scene);
  }
  if (!shadowRT2) {
    shadowRT2 = createShadowRT('shadowRT2', scene);
  }

  applyMaterialUniforms();

  // Bounding box for camera positioning and shadow VP
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  const bbG = segbinData.geoms;
  const bbRoles = segbinData.roles;
  for (let i = 0; i < segbinData.count; i++) {
    if (HIDDEN_ROLES.has(bbRoles[i])) continue;
    const sx = bbG[i * 8], sy = bbG[i * 8 + 1], sz = bbG[i * 8 + 2];
    const ex = bbG[i * 8 + 3], ey = bbG[i * 8 + 4], ez = bbG[i * 8 + 5];
    if (sx < minX) minX = sx; if (sx > maxX) maxX = sx;
    if (sy < minY) minY = sy; if (sy > maxY) maxY = sy;
    if (sz < minZ) minZ = sz; if (sz > maxZ) maxZ = sz;
    if (ex < minX) minX = ex; if (ex > maxX) maxX = ex;
    if (ey < minY) minY = ey; if (ey > maxY) maxY = ey;
    if (ez < minZ) minZ = ez; if (ez > maxZ) maxZ = ez;
  }
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2, cz = (minZ + maxZ) / 2;
  const maxDim = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 1);
  modelCenter = new Vector3(cx, cy, cz);
  modelExtent = maxDim;

  camera.setTarget(modelCenter);
  camera.alpha = Math.PI / 4;
  camera.beta = Math.PI / 4;
  camera.radius = maxDim * 1.5;
  camera.lowerRadiusLimit = maxDim * 0.01;
  camera.upperRadiusLimit = maxDim * 10;


  const lightHeight = maxZ + maxDim * 0.8;
  sunLight.position = new Vector3(cx + maxDim * 0.3, cy - maxDim * 0.3, lightHeight);
  sunLight.direction = new Vector3(cx, cy, cz)
    .subtract(sunLight.position)
    .normalize();

  // Position fill light
  if (fillLight) {
    const fillPos = new Vector3(cx - maxDim * 0.5, cy + maxDim * 0.25, cz - maxDim * 0.3);
    fillLight.position = fillPos;
    fillLight.direction = modelCenter.subtract(fillPos).normalize();
    fillLight.intensity = props.fillLightIntensity ?? 0.4;
  }

  // Lower ground plane to just below model
  if (groundMesh) {
    groundMesh.position.z = minZ - 5;
  }

  // Set up TAA post-process (one-time)
  if (!taaPost) {
    const w = engine.getRenderWidth();
    const h = engine.getRenderHeight();

    // Register TAA pixel shader with Babylon's shader store
    Effect.ShadersStore["taaPixelShader"] = TAA_PIXEL_SHADER;

    // History texture (initialised to black)
    taaHistoryTex = new RawTexture(new Float32Array(w * h * 4), w, h,
      Engine.TEXTUREFORMAT_RGBA, scene, false, false,
      Texture.NEAREST_SAMPLINGMODE, Engine.TEXTURETYPE_HALF_FLOAT);
    taaHistoryTex.wrapU = Texture.CLAMP_ADDRESSMODE;
    taaHistoryTex.wrapV = Texture.CLAMP_ADDRESSMODE;

    // TAA post-process (uses Babylon's default vertex shader, reusable for stable output)
    const pp = new PostProcess("taa", "taa",
      ["uBlendFactor", "uScreenSize"],
      ["uHistoryTex"],
      { width: w, height: h }, camera, Texture.NEAREST_SAMPLINGMODE,
      engine, true // reusable=true → output texture stable between frames
    );
    pp.onApplyObservable.add((effect) => {
      effect.setTexture("uHistoryTex", taaHistoryTex!);
      effect.setFloat("uBlendFactor", 0.1);
      effect.setFloat2("uScreenSize", engine.getRenderWidth(), engine.getRenderHeight());
    });
    // After render, save output as next frame's history reference
    // (reusable=true keeps the double-buffered texture valid between frames)
    pp.onAfterRenderObservable.add(() => {
      if (taaPost) {
        const internalTex = taaPost.outputTexture;
        if (internalTex && taaHistoryTex) {
          taaHistoryTex.dispose();
          const wrapper = new Texture(null, scene);
          wrapper._texture = internalTex;
          wrapper.wrapU = Texture.CLAMP_ADDRESSMODE;
          wrapper.wrapV = Texture.CLAMP_ADDRESSMODE;
          taaHistoryTex = wrapper;
        }
      }
    });
    taaPost = pp;
    log('Custom TAA post-process ready');
  }

  const totalMs = performance.now() - t0;
  log(`Segbin loaded in ${totalMs.toFixed(0)}ms`);
  emit('model-loaded', Math.round(totalMs));
}

function applyMaterialUniforms() {
  for (const m of segbinMeshes) {
    const mat = m.material as ShaderMaterial;
    mat.setFloat('uRoughness', props.roughness);
    mat.setFloat('uMetalness', props.metalness);
    mat.setFloat('uEnvIntensity', props.envIntensity);
    mat.setFloat('uSpecularStrength', props.specularStrength);
    mat.setFloat('uAmbientStrength', props.ambientStrength);
    mat.setColor3('uBaseColorTint', Color3.FromHexString(props.baseColorTint));
    if (envMapTexture) {
      mat.setTexture('uEnvMapEQ', envMapTexture);
      mat.setFloat('uEnvMapLOD', 8.0);
    }
    mat.setFloat('uShadowSoftness', props.shadowSoftness ?? 2.0);
    mat.setFloat('uUseRoleColors', props.roleColors !== false ? 1.0 : 0.0);
    mat.setFloat('uKeyLightIntensity', props.keyLightIntensity ?? 1.0);
    mat.setFloat('uFillLightIntensity', props.fillLightIntensity ?? 0.4);
  }
}

// ── Shadow map helpers ──────────────────────────────────────────────────

function createShadowRT(name: string, scene: Scene): RenderTargetTexture {
  const rt = new RenderTargetTexture(name, 2048, scene, false, true, Engine.TEXTURETYPE_FLOAT);
  rt.wrapU = Texture.CLAMP_ADDRESSMODE;
  rt.wrapV = Texture.CLAMP_ADDRESSMODE;
  return rt;
}

function computeLightVP(lightDir: Vector3, target: Vector3, extent: number): Matrix {
  // Place the "camera" opposite the light direction
  const lightPos = target.clone().add(lightDir.scale(-extent * 2));
  const view = Matrix.LookAtLH(lightPos, target, Vector3.Up());
  const proj = Matrix.OrthoOffCenterLH(-extent, extent, -extent, extent, 0, extent * 4);
  return view.multiply(proj);
}

function renderShadowMap(buildResult: BuildResult, rt: RenderTargetTexture, lightVP: Matrix) {
  // Swap to shadow material, render, restore
  const mesh = buildResult.shadowMesh;
  const origMat = mesh.material;
  mesh.material = buildResult.shadowMat;
  buildResult.shadowMat.setMatrix('uLightVP', lightVP);
  // Ensure shadow RT uses the mesh
  if (!rt.renderList || rt.renderList.indexOf(mesh) < 0) {
    rt.renderList = [mesh];
  }
  rt.render();
  mesh.material = origMat;
}

// ── Halton sequence for TAA sub-pixel jitter ──

function halton2(i: number): number {
  let f = 1, r = 0;
  while (i > 0) { f /= 2; r += (i & 1) * f; i >>= 1; }
  return r - 0.5;
}
function halton3(i: number): number {
  let f = 1, r = 0;
  while (i > 0) { f /= 3; r += (i % 3) * f; i = Math.floor(i / 3); }
  return r - 0.5;
}

function renderFrame() {
  if (disposed) return;
  if (!buildResult) { scene.render(); return; }

  // Per-segment LOD
  if (segbinData && screenshotLodLock < 0) {
    const g = segbinData.geoms;
    const st = segbinData.segType;
    const vpH = engine.getRenderHeight();
    const fov = camera.fov;
    const tanFovHalf = Math.tan(fov * 0.5);
    const viewMat = camera.getViewMatrix();
    const mp = new Vector3();
    const vp = new Vector3();
    const segLod = new Uint8Array(segbinData.count);

    for (let i = 0; i < segbinData.count; i++) {
      const sx = g[i * 8], sy = g[i * 8 + 1], sz = g[i * 8 + 2];
      const ex = g[i * 8 + 3], ey = g[i * 8 + 4], ez = g[i * 8 + 5];
      const w = g[i * 8 + 6];
      const dx = ex - sx, dy = ey - sy, dz = ez - sz;
      const segLen = Math.sqrt(dx * dx + dy * dy + dz * dz);
      mp.set((sx + ex) * 0.5, (sy + ey) * 0.5, (sz + ez) * 0.5);
      Vector3.TransformCoordinatesToRef(mp, viewMat, vp);
      const depth = Math.abs(vp.z);
      const screenSize = Math.max(segLen, w) * vpH / (2.0 * depth * tanFovHalf);
      let lod = 0;
      if (screenSize < 6.0) lod = 2;
      else if (screenSize < 24.0) lod = 1;
      if (st[i] === 1 && i > 0) lod = segLod[i - 1];
      segLod[i] = lod;
      for (let li = 0; li < 3; li++) {
        buildResult.groups[li].bodyVis[i * 4] = (buildResult.groups[li].lod === lod) ? 1.0 : 0.0;
      }
    }
    for (let li = 0; li < 2; li++) {
      const g2 = buildResult.groups[li];
      if (!g2.capsVis || !g2.capSegIdx) continue;
      for (let ci = 0; ci < g2.capSegIdx.length; ci++) {
        g2.capsVis[ci * 4] = g2.bodyVis[g2.capSegIdx[ci] * 4];
      }
    }
    for (const g2 of buildResult.groups) {
      g2.body.thinInstanceSetBuffer('instVisible', g2.bodyVis, 4);
      if (g2.caps) g2.caps.thinInstanceSetBuffer('instVisible', g2.capsVis!, 4);
    }

    // Triangle count
    let triSum = 0;
    for (let li = 0; li < 3; li++) {
      const g2 = buildResult.groups[li];
      const bodyTris = LOD_BODY_GEO[li].indices.length / 3;
      let n = 0;
      for (let i = 0; i < segbinData.count; i++) {
        if (g2.bodyVis[i * 4] > 0.5) n++;
      }
      triSum += bodyTris * n;
      if (g2.capsVis) {
        const capTris = LOD_CAP_GEO[li].indices.length / 3;
        let cn = 0;
        for (let ci = 0; ci < g2.capsVis.length / 4; ci++) {
          if (g2.capsVis[ci * 4] > 0.5) cn++;
        }
        triSum += capTris * cn;
      }
    }
    lastTriCount = triSum;
  }


  // ── Render shadow maps ──
  if (shadowRT && buildResult.shadowMat) {
    const shadowTarget = modelCenter;
    const shadowExtent = modelExtent * 1.5;

    // Light 1 (key) — Babylon's sunLight.direction points FROM the light
    // TOWARD the target (light travel direction). For the shader's NdotL we
    // need the direction TOWARD the light, so we negate.
    const lightDir1 = sunLight.direction.clone().normalize();
    const vp1 = computeLightVP(lightDir1, shadowTarget, shadowExtent);
    renderShadowMap(buildResult, shadowRT, vp1);

    // Light 2 (fill)
    let vp2: Matrix | null = null;
    if (shadowRT2 && fillLight) {
      const lightDir2 = fillLight.direction.clone().normalize();
      vp2 = computeLightVP(lightDir2, shadowTarget, shadowExtent);
      renderShadowMap(buildResult, shadowRT2, vp2);
    }

    // Set uniforms on main materials (key light always set)
    const camPos = camera.position;
    const shadowMapSize = new Vector2(2048, 2048);
    const keyDir = lightDir1.negate(); // toward the light for shader
    // Fill light direction toward the light (decoupled from shadow map)
    let fillDir: Vector3 | null = null;
    if (fillLight) {
      fillDir = fillLight.direction.clone().negate().normalize();
    }
    for (const m of segbinMeshes) {
      const mat = m.material as ShaderMaterial;
      mat.setVector3('uCameraPos', camPos);
      mat.setVector3('uKeyLightDir', keyDir);
      mat.setTexture('uShadowMap', shadowRT);
      mat.setMatrix('uShadowMatrix', vp1);
      mat.setVector2('uShadowMapSize', shadowMapSize);
      // Fill light — direction always set if fillLight exists, shadow optional
      if (fillDir) {
        mat.setVector3('uKeyLightDir2', fillDir);
      }
      if (vp2 && fillLight) {
        mat.setTexture('uShadowMap2', shadowRT2);
        mat.setMatrix('uShadowMatrix2', vp2);
        mat.setVector2('uShadowMapSize2', shadowMapSize);
      }
    }
    // Update ground plane shadow uniforms (key light only)
    if (groundMat) {
      groundMat.setTexture('uShadowMap', shadowRT);
      groundMat.setMatrix('uShadowMatrix', vp1);
      groundMat.setVector2('uShadowMapSize', shadowMapSize);
      groundMat.setFloat('uShadowSoftness', props.shadowSoftness ?? 2.0);
    }
  }

  // ── TAA: apply Halton jitter to camera projection ──
  if (taaPost) {
    const w = engine.getRenderWidth();
    const h = engine.getRenderHeight();
    const jx = halton2(taaFrame);
    const jy = halton3(taaFrame);
    const origProj = camera.getProjectionMatrix().clone();
    const proj = camera.getProjectionMatrix();
    proj.m[8] += (jx * 2) / w;
    proj.m[9] += (jy * 2) / h;
    scene.render();
    // Restore original projection for next frame's fresh jitter
    camera.getProjectionMatrix().copyFrom(origProj);
    taaFrame++;
  } else {
    scene.render();
  }

  // FPS
  statsFrames++;
  const now = performance.now();
  if (now - statsTime >= 1000) {
    stats.value = {
      show: stats.value.show,
      fps: Math.round(statsFrames / ((now - statsTime) / 1000)),
      triangles: Math.round(lastTriCount / 1000),
    };
    statsFrames = 0;
    statsTime = now;
  }
}

onMounted(() => {
  if (props.rendererType === 'webgpu') {
    log('Mounting WebGPU renderer...');
    const canvas = canvasEl.value!;
    const cont = container.value!;

    import('@sliced/webgpu-renderer').then(async ({ WebGPURenderer }) => {
      try {
        const renderer = new WebGPURenderer();
        await renderer.mount(cont, canvas);
        webgpuRenderer = renderer;

        // Stats keyboard toggle
        const onKey = (e: KeyboardEvent) => {
          if (e.key === '`') stats.value.show = !stats.value.show;
        };
        window.addEventListener('keydown', onKey);

        // Poll stats from the WebGPU renderer
        const statsTimer = setInterval(() => {
          if (webgpuRenderer) {
            stats.value = { ...webgpuRenderer.stats, show: stats.value.show };
          }
        }, 200);
        webgpuStatsTimer = statsTimer;

        renderer.setMaterial({
          roughness: props.roughness,
          metalness: props.metalness,
          envIntensity: props.envIntensity,
          specularStrength: props.specularStrength,
          ambientStrength: props.ambientStrength,
          baseColorTint: props.baseColorTint,
        });
        if (renderer.ssaoEnabled !== undefined) {
          renderer.ssaoEnabled = props.ssaoEnabled !== false;
        }
        renderer.pipeline.material.useRoleColors = props.roleColors !== false ? 1 : 0;
        renderer.pipeline.writeMaterialUBO();
        if (props.shadowSoftness !== undefined) renderer.setShadowSoftness?.(props.shadowSoftness);
        if (props.keyLightIntensity !== undefined) renderer.setKeyLightIntensity?.(props.keyLightIntensity);
        if (props.fillLightIntensity !== undefined) renderer.setFillLightIntensity?.(props.fillLightIntensity);
        if (props.contactShadowDist !== undefined) renderer.pipeline.contactShadowDist = props.contactShadowDist;
        if (props.contactShadowStrength !== undefined) renderer.setContactShadowStrength?.(props.contactShadowStrength);
        if (props.ssaoIntensity !== undefined) renderer.setSSAOIntensity?.(props.ssaoIntensity);
        if (props.ssaoRadius !== undefined) renderer.setSSAORadius?.(props.ssaoRadius);

        if (props.segbinUrl) {
          const ms = await renderer.loadModel(props.segbinUrl);
          emit('model-loaded', ms);
        }

        log('WebGPU renderer ready');
      } catch (err) {
        console.error('[ModelViewer] WebGPU mount failed:', err);
      }
    }).catch((err) => {
      console.error('[ModelViewer] WebGPU module failed to load:', err);
    });
    return;
  }

  log('Mounting, setting up Babylon.js...');
  const t0 = performance.now();

  try {
    const canvas = canvasEl.value!;
    engine = new Engine(canvas, true, {
      preserveDrawingBuffer: false, stencil: false,
    });
    scene = new Scene(engine);
    scene.clearColor = new Color4(0.15, 0.15, 0.17, 1.0);

    camera = new ArcRotateCamera('camera', Math.PI / 4, Math.PI / 4, 150,
      new Vector3(0, 0, 0), scene);
    camera.upVector = new Vector3(0, 0, 1);
    camera.upperBetaLimit = Math.PI;
    camera.lowerBetaLimit = 0.01;
    camera.attachControl(true);

    // Post-processing — custom velocity TAA (replaces Babylon TAA+FXAA)
    // Depth renderer for velocity pass (stores non-linear depth for reprojection)
    const dr = scene.enableDepthRenderer(camera, true, false, true);
    depthRenderer = dr;



    // Lighting
    const hemi = new HemisphericLight('hemi', new Vector3(0, 0, 1), scene);
    hemi.intensity = 0.3;
    sunLight = new DirectionalLight('sun', new Vector3(-0.416, 0.25, -0.872), scene);
    sunLight.intensity = 4.0;

    // Second directional light (fill)
    fillLight = new DirectionalLight('fill', new Vector3(0.5, -0.25, 0.3), scene);
    fillLight.intensity = 1.0;

    // Ground plane
    groundMesh = MeshBuilder.CreateGround('ground', { width: 500, height: 500 }, scene);
    const groundVertSrc = `
      precision highp float;
      in vec3 position;
      uniform mat4 worldViewProjection;
      uniform mat4 uShadowMatrix;
      out vec3 vWorldPos;
      void main() {
        vWorldPos = position;
        gl_Position = worldViewProjection * vec4(position, 1.0);
      }
    `;
    const groundFragSrc = `
      precision highp float;
      uniform sampler2D uShadowMap;
      uniform mat4 uShadowMatrix;
      uniform vec2 uShadowMapSize;
      uniform float uShadowSoftness;
      in vec3 vWorldPos;
      layout(location = 0) out vec4 fragColor;

      float interleavedGradientNoise(vec2 pos) {
        return fract(52.9829189 * fract(dot(pos, vec2(0.06711056, 0.00583715))));
      }
      vec2 vogelDiskSample(int index, int count, float phi) {
        float goldenAngle = 2.399963229728653;
        float r = sqrt((float(index) + 0.5) / float(count));
        float theta = float(index) * goldenAngle + phi;
        return vec2(cos(theta), sin(theta)) * r;
      }
      float shadowFactor() {
        vec4 p = uShadowMatrix * vec4(vWorldPos, 1.0);
        p.xyz /= p.w;
        p.xyz = p.xyz * 0.5 + 0.5;
        if (p.x < 0.0 || p.x > 1.0 || p.y < 0.0 || p.y > 1.0 || p.z > 1.0 || p.z < 0.0) return 1.0;
        float texelSize = 1.0 / uShadowMapSize.x;
        float radius = texelSize * uShadowSoftness;
        float phi = interleavedGradientNoise(gl_FragCoord.xy) * 6.283185307;
        float sum = 0.0;
        for (int i = 0; i < 8; i++) {
          vec2 offset = vogelDiskSample(i, 8, phi) * radius;
          vec2 uv = clamp(p.xy + offset, vec2(0.0), vec2(1.0));
          float d = texture(uShadowMap, uv).r;
          sum += (p.z > d) ? 0.0 : 1.0;
        }
        return sum / 8.0;
      }
      void main() {
        float sf = shadowFactor();
        vec3 groundColor = vec3(0.12, 0.12, 0.14);
        vec3 lit = mix(groundColor * 0.3, groundColor * 1.0, sf);
        fragColor = vec4(lit, 1.0);
      }
    `;
    const groundMatRef = new ShaderMaterial('groundMat', scene, {
      vertexSource: groundVertSrc,
      fragmentSource: groundFragSrc,
    }, {
      attributes: ['position'],
      uniforms: ['worldViewProjection', 'uShadowMatrix', 'uShadowMapSize', 'uShadowSoftness'],
      samplers: ['uShadowMap'],
      needAlphaBlending: false,
    });
    groundMatRef.backFaceCulling = false;
    groundMesh.material = groundMatRef;
    groundMat = groundMatRef;
    groundMesh.position = new Vector3(0, 0, -10);
    groundMesh.isPickable = false;

    // SSAO (ambient occlusion) — NOTE: skipped for custom ShaderMaterial meshes
    // Babylon's SSAO2RenderingPipeline uses an internal depth renderer that
    // doesn't inherit our custom vertex shader (segment position computation from
    // instance data). It would compute wrong world positions. A custom SSAO
    // pass similar to the WebGPU version would be needed for correct results.
    // For now, the PBR lighting + shadows provide adequate depth cues.

    loadEnvMap();
    log(`Setup complete in ${(performance.now() - t0).toFixed(0)}ms`);

    // Controls
    window.addEventListener('keydown', (e) => {
      if (e.key === '`') { stats.value.show = !stats.value.show; }
    });
    statsTime = performance.now();

    engine.runRenderLoop(renderFrame);

    if (props.segbinUrl) {
      loadSegbinModel(props.segbinUrl);
    } else {
      log('onMounted: segbinUrl is null, waiting');
    }
  } catch (_err) {
    console.error('[ModelViewer] Error:', _err);
  }
});

watch(
  () => props.segbinUrl,
  (newUrl, oldUrl) => {
    if (!newUrl || disposed) return;
    if (newUrl === oldUrl) return;
    if (webgpuRenderer) {
      webgpuRenderer.loadModel(newUrl).then((ms) => emit('model-loaded', ms));
      return;
    }
    // WebGPU selected but not loaded yet — fall through to Babylon if it was initialized
    if (props.rendererType === 'webgpu') {
      log('segbinUrl change before WebGPU renderer ready — deferring');
      return;
    }
    loadSegbinModel(newUrl);
  },
);

onBeforeUnmount(() => {
  log('Disposing');
  disposed = true;
  resizeObserver?.disconnect();
  if (webgpuRenderer) { webgpuRenderer.dispose(); webgpuRenderer = null; clearInterval(webgpuStatsTimer); return; }
  engine.stopRenderLoop();
  shadowRT?.dispose();
  shadowRT2?.dispose();
  shadowRT = null;
  shadowRT2 = null;
  taaPost?.dispose();
  taaPost = null;
  taaHistoryTex?.dispose();
  taaHistoryTex = null;
  for (const m of segbinMeshes) {
    m.dispose();
  }
  segbinMeshes = [];
  groundMesh?.dispose();
  groundMesh = null;
  scene?.dispose();
  engine?.dispose();
});

// Debug hooks for headless screenshot testing
if (typeof location !== 'undefined' && location.search.includes('screenshot')) {
  const scr: Record<string, any> = {};
  scr.forceLOD = (lod: number) => {
    if (!buildResult) return;
    screenshotLodLock = lod;
    for (const g of buildResult.groups) {
      const v = g.lod === lod ? 1.0 : 0.0;
      g.bodyVis.fill(v, 0, g.bodyVis.length);
      g.body.thinInstanceSetBuffer('instVisible', g.bodyVis, 4);
    }
    const activeGroup = buildResult.groups[lod];
    for (let li = 0; li < 2; li++) {
      const g = buildResult.groups[li];
      if (!g.capsVis || !g.capSegIdx) continue;
      for (let ci = 0; ci < g.capSegIdx.length; ci++) {
        g.capsVis[ci * 4] = activeGroup.bodyVis[g.capSegIdx[ci] * 4];
      }
      g.caps!.thinInstanceSetBuffer('instVisible', g.capsVis, 4);
    }
  };
  scr.getTriCount = () => lastTriCount;
  (window as any).__screenshot = scr;
}
</script>

<style scoped>
.preview-container {
  width: 100%;
  height: 100%;
  position: relative;
  overflow: hidden;
}

.render-canvas {
  width: 100%;
  height: 100%;
  display: block;
  outline: none;
  touch-action: none;
}

.stats-overlay {
  position: absolute;
  top: 8px;
  left: 8px;
  z-index: 10;
  color: #0f0;
  font: 12px/1.5 monospace;
  background: rgba(0, 0, 0, 0.6);
  padding: 6px 8px;
  border-radius: 4px;
  pointer-events: none;
}
</style>
