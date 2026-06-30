<template>
  <div ref="container" class="preview-container">
    <canvas ref="canvasEl" class="render-canvas" />
  </div>
  <div class="stats-overlay" v-if="stats.show">FPS {{ stats.fps }}<br>tris {{ stats.triangles }}k
    <br>GPU: {{ stats.gpuMs.offscreen > 0 ? fmtMs(stats.gpuMs.offscreen) : '--' }} · SSAO {{ stats.gpuMs.ssao > 0 ? fmtMs(stats.gpuMs.ssao) : '--' }} · blur {{ stats.gpuMs.blur > 0 ? fmtMs(stats.gpuMs.blur) : '--' }} · vel {{ stats.gpuMs.velocity > 0 ? fmtMs(stats.gpuMs.velocity) : '--' }} · frame {{ fmtMs(stats.gpuMs.frameMs) }}
    <br><label class="keepalive-label"><input type="checkbox" v-model="keepAliveLocal" /> Keep alive</label>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, watch, onBeforeUnmount } from 'vue';

const container = ref<HTMLDivElement>();
const canvasEl = ref<HTMLCanvasElement>();

const props = withDefaults(defineProps<{
  segbinUrl: string | null;
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
  arcCurvature?: number;
  /** Debug preview mode for WebGPU renderer. */
  debugPreview?: 'none' | 'depth' | 'occlusion' | 'color' | 'normal' | 'shadow' | 'shadow2' | 'velocity' | 'composite-taa' | 'blur-temp' | 'brdf-lut' | 'prefilter-up' | 'prefilter-fwd' | 'prefilter-down' | 'source-up' | 'source-fwd' | 'source-down' | 'worldpos';
  envMapUrl?: string;
  keepAlive?: boolean;
}>(), {
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
  ssaoIntensity: 0.5,
  ssaoRadius: 0.5,
  arcCurvature: 1.0,
});
const emit = defineEmits<{ 'model-loaded': [ms: number] }>();

const stats = ref<{ show: boolean; fps: number; triangles: number; gpuMs: { offscreen: number; ssao: number; blur: number; velocity: number; frameMs: number } }>({ show: true, fps: 0, triangles: 0, gpuMs: { offscreen: 0, ssao: 0, blur: 0, velocity: 0, frameMs: 0 } });

/** Local copy of keepAlive so the checkbox is reactive even before the renderer exists. */
const keepAliveLocal = ref(!!props.keepAlive);
watch(keepAliveLocal, (v) => { if (renderer) renderer.setKeepAlive?.(v); });

function fmtMs(ms: number): string {
  if (ms < 0.1) return '<0.1';
  if (ms < 10) return ms.toFixed(1);
  return ms.toFixed(0);
}

let renderer: any = null;
let statsTimer: any = null;

function log(msg: string, ...args: any[]) {
  console.log(`[ModelViewer] ${msg}`, ...args);
}

// ── Material props → WebGPU renderer ──

function onMaterialChange() {
  if (!renderer) return;
  renderer.setMaterial({
    roughness: props.roughness,
    metalness: props.metalness,
    envIntensity: props.envIntensity,
    specularStrength: props.specularStrength,
    ambientStrength: props.ambientStrength,
    baseColorTint: props.baseColorTint,
    ssaoIntensity: props.ssaoIntensity,
    ssaoRadius: props.ssaoRadius,
    arcCurvature: props.arcCurvature,
  });
  renderer.setSSAOIntensity?.(props.ssaoIntensity);
  renderer.setSSAORadius?.(props.ssaoRadius);
  renderer.setArcCurvature?.(props.arcCurvature);
  renderer.setShadowSoftness?.(props.shadowSoftness);
  renderer.setKeyLightIntensity?.(props.keyLightIntensity);
  renderer.setFillLightIntensity?.(props.fillLightIntensity);
  if (renderer.ssaoEnabled !== undefined) renderer.ssaoEnabled = props.ssaoEnabled;
}

watch(() => [
  props.roughness, props.metalness, props.envIntensity, props.specularStrength,
  props.ambientStrength, props.baseColorTint, props.ssaoEnabled, props.roleColors,
  props.shadowSoftness, props.keyLightIntensity, props.fillLightIntensity,
  props.contactShadowDist, props.contactShadowStrength, props.ssaoIntensity, props.ssaoRadius,
  props.arcCurvature,
], onMaterialChange);

// ── Debug preview ──

watch(() => props.debugPreview, (v) => {
  if (renderer) renderer.debugPreview = v ?? 'none';
}, { immediate: true });

// ── Env map ──

watch(() => props.envMapUrl, (url) => {
  if (renderer && typeof renderer.setEnvMap === 'function') {
    renderer.setEnvMap(url);
  }
});

watch(() => props.keepAlive, (v) => {
  if (renderer) renderer.setKeepAlive?.(!!v);
});

// ── Resize ──

function resize() {
  if (renderer && container.value) {
    // Pass container dimensions directly so the renderer doesn't
    // read stale CSS sizes from the canvas (which has fixed style values
    // from a previous resize).
    renderer.resize(container.value.clientWidth, container.value.clientHeight);
  }
}

let resizeObserver: ResizeObserver | null = null;

// ── Mount ──

onMounted(async () => {
  if (!container.value) return;
  resizeObserver = new ResizeObserver(() => resize());
  resizeObserver.observe(container.value);

  log('Detecting renderer capability...');
  const canvas = canvasEl.value!;
  const cont = container.value!;

  // ── Try WebGPU first, fall back to PlayCanvas (WebGL2) ──
  let r: any;
  if (navigator.gpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) {
        log('WebGPU available — using WebGPU renderer');
        const { WebGPURenderer } = await import('@sliced/webgpu-renderer');
        r = new WebGPURenderer();
        await r.mount(cont, canvas);
      }
    } catch (e) {
      log(`WebGPU init failed: ${e}`);
    }
  }
  if (!r) {
    log('Falling back to PlayCanvas (WebGL2)');
    const { PlayCanvasRenderer } = await import('@sliced/playcanvas-renderer');
    r = new PlayCanvasRenderer();
    await r.mount(cont, canvas);
  }
  renderer = r;
  await setupRenderer(r);
  log('Renderer ready');

  async function setupRenderer(r: any) {
    // Stats keyboard toggle
    const onKey = (e: KeyboardEvent) => {
      if (e.key === '`') stats.value.show = !stats.value.show;
    };
    window.addEventListener('keydown', onKey);

    // Poll stats
    statsTimer = setInterval(() => {
      if (renderer) {
        stats.value = { ...renderer.stats, show: stats.value.show };
      }
    }, 200);

    // Apply initial material & lighting settings (uses the same
    // comprehensive setter as the watcher — sets key/fill light
    // intensity, shadow softness, SSAO params, etc.)
    onMaterialChange();
    r.setKeepAlive?.(keepAliveLocal.value);

    if (props.segbinUrl) {
      const ms = await r.loadModel(props.segbinUrl);
      emit('model-loaded', ms);
    }
  }
});

// ── Model URL changes ──

watch(
  () => props.segbinUrl,
  (newUrl, oldUrl) => {
    if (!newUrl) return;
    if (newUrl === oldUrl) return;
    if (renderer) {
      renderer.loadModel(newUrl).then((ms: number) => emit('model-loaded', ms));
    }
  },
);

// ── Dispose ──

onBeforeUnmount(() => {
  log('Disposing');
  resizeObserver?.disconnect();
  if (renderer) { renderer.dispose(); renderer = null; }
  if (statsTimer) clearInterval(statsTimer);
});
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
.stats-overlay .keepalive-label {
  pointer-events: auto;
  cursor: pointer;
  color: #aaa;
  font-size: 11px;
}
.stats-overlay .keepalive-label input {
  vertical-align: middle;
}
</style>
