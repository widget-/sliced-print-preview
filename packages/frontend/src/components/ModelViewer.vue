<template>
  <div ref="container" class="preview-container">
    <canvas ref="canvasEl" class="render-canvas" />
  </div>
  <div class="stats-overlay" v-if="stats.show">FPS {{ stats.fps }}<br>tris {{ stats.triangles }}k</div>
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

const stats = ref({ show: false, fps: 0, triangles: 0 });

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

// ── Resize ──

function resize() {
  if (renderer) renderer.resize();
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

  // PlayCanvas handles WebGPU → WebGL2 fallback internally
  const { PlayCanvasRenderer } = await import('@sliced/playcanvas-renderer');
  const r = new PlayCanvasRenderer();
  await r.mount(cont, canvas);
  renderer = r;
  await setupRenderer(r);
  log('PlayCanvas renderer ready');

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

    // Apply initial material settings
    r.setMaterial({
      roughness: props.roughness,
      metalness: props.metalness,
      envIntensity: props.envIntensity,
      specularStrength: props.specularStrength,
      ambientStrength: props.ambientStrength,
      baseColorTint: props.baseColorTint,
    });

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
</style>
