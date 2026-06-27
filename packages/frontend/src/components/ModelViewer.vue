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
import { Vector3 } from '@babylonjs/core/Maths/math.vector';
import { Color4, Color3 } from '@babylonjs/core/Maths/math.color';
import { ShaderMaterial } from '@babylonjs/core/Materials/shaderMaterial';
import { RawTexture } from '@babylonjs/core/Materials/Textures/rawTexture';
import { Texture } from '@babylonjs/core/Materials/Textures/texture';
import { HDRTools } from '@babylonjs/core/Misc/HighDynamicRange/hdr';
import { TAARenderingPipeline } from '@babylonjs/core/PostProcesses/RenderPipeline/Pipelines/taaRenderingPipeline';
import { FxaaPostProcess } from '@babylonjs/core/PostProcesses/fxaaPostProcess';

import { parseSegbin, buildShaderMeshes, Role } from '../renderer/SegbinLoader';
import type { SegbinData } from '@sliced/shared';
import type { BuildResult } from '../renderer/SegbinLoader';
import { LOD_BODY_GEO, LOD_CAP_GEO } from '../renderer/geometry';


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
    mat.setColor3('uBaseColorTint', Color3.FromHexString(props.baseColorTint));
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
let taaPipeline: TAARenderingPipeline | null = null;
let fxaaProcess: FxaaPostProcess | null = null;

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

  applyMaterialUniforms();

  // Bounding box for camera positioning
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  const g = segbinData.geoms;
  const roles = segbinData.roles;
  for (let i = 0; i < segbinData.count; i++) {
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
  const maxDim = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 1);

  camera.setTarget(new Vector3(cx, cy, cz));
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
    mat.setColor3('uBaseColorTint', Color3.FromHexString(props.baseColorTint));
    if (envMapTexture) {
      mat.setTexture('uEnvMapEQ', envMapTexture);
      mat.setFloat('uEnvMapLOD', 8.0);
    }
  }
}

function renderFrame() {
  if (disposed) return;

  // Per-segment LOD
  if (buildResult && segbinData && screenshotLodLock < 0) {
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

  // Camera uniforms
  const camPos = camera.position;
  for (const m of segbinMeshes) {
    (m.material as ShaderMaterial).setVector3('uCameraPos', camPos);
  }

  scene.render();

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

    // Post-processing
    const taa = new TAARenderingPipeline('taa', scene, [camera], 2);
    taa.samples = 8;
    taa.reprojectHistory = false;
    taa.disableOnCameraMove = true;
    taaPipeline = taa;

    const fxaa = new FxaaPostProcess('fxaa', 1.0, camera);
    fxaaProcess = fxaa;


    // Lighting
    const hemi = new HemisphericLight('hemi', new Vector3(0, 0, 1), scene);
    hemi.intensity = 0.3;
    sunLight = new DirectionalLight('sun', new Vector3(-0.416, 0.25, -0.872), scene);
    sunLight.intensity = 4.0;

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
  taaPipeline?.dispose();
  fxaaProcess?.dispose();
  for (const m of segbinMeshes) {
    m.dispose();
  }
  segbinMeshes = [];
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
