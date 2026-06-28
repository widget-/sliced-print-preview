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
  ssaoIntensity: 3.0,
  ssaoRadius: 0.06,
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
  });
  if (renderer.ssaoEnabled !== undefined) {
    renderer.ssaoEnabled = props.ssaoEnabled !== false;
  }
  // WebGPU-specific controls (safe-checked)
  if (renderer.pipeline?.material?.useRoleColors !== undefined) {
    renderer.pipeline.material.useRoleColors = props.roleColors !== false ? 1 : 0;
    renderer.pipeline.writeMaterialUBO?.();
  }
  renderer.setShadowSoftness?.(props.shadowSoftness ?? 2.0);
  renderer.setKeyLightIntensity?.(props.keyLightIntensity ?? 1.0);
  renderer.setFillLightIntensity?.(props.fillLightIntensity ?? 0.4);
  if (renderer.pipeline?.contactShadowDist !== undefined) {
    renderer.pipeline.contactShadowDist = props.contactShadowDist;
  }
  renderer.setContactShadowStrength?.(props.contactShadowStrength ?? 1.0);
  renderer.setSSAOIntensity?.(props.ssaoIntensity ?? 0.35);
  renderer.setSSAORadius?.(props.ssaoRadius ?? 0.06);
}

watch(() => [
  props.roughness, props.metalness, props.envIntensity, props.specularStrength,
  props.ambientStrength, props.baseColorTint, props.ssaoEnabled, props.roleColors,
  props.shadowSoftness, props.keyLightIntensity, props.fillLightIntensity,
  props.contactShadowDist, props.contactShadowStrength, props.ssaoIntensity, props.ssaoRadius,
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

  // Feature-detect: try WebGPU first, fall back to WebGL 2.0
  const hasWebGPU = !!navigator.gpu;

  if (hasWebGPU) {
    // Test if we can actually get a WebGPU adapter
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) {
        await mountWebGPU();
        return;
      }
    } catch { /* fall through to WebGL */ }
    log('WebGPU adapter not available, falling back to WebGL 2.0');
  } else {
    log('WebGPU not available, using WebGL 2.0');
  }

  await mountWebGL();

  async function mountWebGPU() {
    const { WebGPURenderer } = await import('@sliced/webgpu-renderer');
    const r = new WebGPURenderer();
    await r.mount(cont, canvas);
    renderer = r;
    await setupRenderer(r, true);
    log('WebGPU renderer ready');
  }

  async function mountWebGL() {
    const testCanvas = document.createElement('canvas');
    const testGl = testCanvas.getContext('webgl2');
    if (!testGl) {
      console.error('[ModelViewer] WebGL 2.0 not available — no renderer possible');
      return;
    }
    const { WebGLRenderer } = await import('@sliced/webgl-renderer');
    const r = new WebGLRenderer();
    await r.mount(cont, canvas);
    renderer = r;
    await setupRenderer(r, false);
    log('WebGL2 renderer ready');
  }

  async function setupRenderer(r: any, isWebGPU: boolean) {
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

    if (isWebGPU) {
      if (r.ssaoEnabled !== undefined) {
        r.ssaoEnabled = props.ssaoEnabled !== false;
      }
      if (r.pipeline?.material) {
        r.pipeline.material.useRoleColors = props.roleColors !== false ? 1 : 0;
        r.pipeline.writeMaterialUBO();
      }
      if (props.shadowSoftness !== undefined) r.setShadowSoftness?.(props.shadowSoftness);
      if (props.keyLightIntensity !== undefined) r.setKeyLightIntensity?.(props.keyLightIntensity);
      if (props.fillLightIntensity !== undefined) r.setFillLightIntensity?.(props.fillLightIntensity);
      if (props.contactShadowDist !== undefined) r.pipeline.contactShadowDist = props.contactShadowDist;
      if (props.contactShadowStrength !== undefined) r.setContactShadowStrength?.(props.contactShadowStrength);
      if (props.ssaoIntensity !== undefined) r.setSSAOIntensity?.(props.ssaoIntensity);
      if (props.ssaoRadius !== undefined) r.setSSAORadius?.(props.ssaoRadius);
      if (props.debugPreview) r.debugPreview = props.debugPreview;
    } else {
      // WebGL fallback supports fewer features
      if (props.shadowSoftness !== undefined) r.setShadowSoftness?.(props.shadowSoftness);
      if (props.keyLightIntensity !== undefined) r.setKeyLightIntensity?.(props.keyLightIntensity);
    }

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
