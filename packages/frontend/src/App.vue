<template>
  <div class="app">
    <div class="sidebar" :style="{ width: sidebarWidth + 'px' }">
      <h3>3D Print Preview</h3>
      <div class="builtin-buttons">
        <button @click="loadBuiltin('benchy')" :disabled="loading">Benchy</button>
        <button @click="loadBuiltin('calibration-cube')" :disabled="loading">Calibration Cube</button>
      </div>
      <p class="builtin-divider">or upload your own</p>
      <input type="file" @change="uploadModel" accept=".stl" />
      <div v-if="loading" class="status">Processing...</div>
      <div v-if="error" class="status error">{{ error }}</div>
      <div v-if="timing.api" class="status timing">
        <div>API: {{ timing.api }}ms · Model: {{ timing.model || '...' }}ms</div>
        <table v-if="timing.slice" class="timing-table">
          <tbody>
            <tr><td>Slice</td><td class="t-ms">{{ fmt(timing.slice) }}</td><td class="t-pct">{{ pct(timing.slice) }}</td></tr>
            <tr><td>Parse</td><td class="t-ms">{{ fmt(timing.parse) }}</td><td class="t-pct">{{ pct(timing.parse) }}</td></tr>
            <tr><td>Segment BVH</td><td class="t-ms">{{ fmt(timing.segBvh) }}</td><td class="t-pct">{{ pct(timing.segBvh) }}</td></tr>
            <tr><td>Ray cull</td><td class="t-ms">{{ fmt(timing.ray) }}</td><td class="t-pct">{{ pct(timing.ray) }}</td></tr>
            <tr v-if="timing.arc"><td>Arc</td><td class="t-ms">{{ fmt(timing.arc) }}</td><td class="t-pct">{{ pct(timing.arc) }}</td></tr>
            <tr><td>Gap</td><td class="t-ms">{{ fmt(timing.gap) }}</td><td class="t-pct">{{ pct(timing.gap) }}</td></tr>
          </tbody>
        </table>
      </div>
      <div id="material-controls">
        <h4>Render Settings</h4>
        <SliderControl label="Roughness" v-model="roughness" :max="1" :step="0.01" />
        <SliderControl label="Metalness" v-model="metalness" :max="1" :step="0.01" />
        <SliderControl label="Env Intensity" v-model="envIntensity" :max="2" :step="0.01" />
        <label>Environment</label>
        <select v-model="envMapUrl" class="renderer-select">
          <option v-for="f in envMapFiles" :key="f" :value="f">{{ f.replace(/_/g, ' ').replace(/\.hdr$/, '') }}</option>
        </select>
        <SliderControl label="Specular" v-model="specularStrength" :max="1" :step="0.01" />
        <SliderControl label="Ambient" v-model="ambientStrength" :max="2" :step="0.01" />
        <label>Color</label>
        <input type="color" v-model="baseColorTint" class="color-picker" />
        <label class="checkbox-label">
          <input type="checkbox" v-model="ssaoEnabled" /> SSAO
        </label>
        <label>Renderer</label>
        <select v-model="rendererType" class="renderer-select">
          <option value="webgl2">WebGL2</option>
          <option value="webgpu" :disabled="!webgpuAvailable" :title="!webgpuAvailable && webgpuReason ? webgpuReason : ''">WebGPU{{ webgpuAvailable ? '' : ' (unavailable)' }}</option>
        </select>
        <label>Debug Preview</label>
        <select v-model="debugPreview" class="renderer-select">
          <option value="none">None</option>
          <option value="color">Offscreen Color</option>
          <option value="normal">Normals</option>
          <option value="depth">Depth</option>
          <option value="occlusion">SSAO Occlusion</option>
          <option value="blur-temp">SSAO Blur Temp</option>
          <option value="brdf-lut">BRDF LUT</option>
          <option value="prefilter-up">Prefilter +Z (sky)</option>
          <option value="prefilter-fwd">Prefilter +Y (north)</option>
          <option value="prefilter-down">Prefilter -Z (ground)</option>
          <option value="source-up">Source +Z (sky)</option>
          <option value="source-fwd">Source +Y (north)</option>
          <option value="source-down">Source -Z (ground)</option>
          <option value="shadow">Shadow Map</option>
          <option value="velocity">Velocity</option>
          <option value="composite-taa">Composite (TAA input)</option>
        </select>
      </div>
    </div>
    <div class="resize-handle" @mousedown="startResize" @touchstart.prevent="startResize" />
    <div class="viewer" :style="isMobile() ? { height: viewerPercent + '%' } : undefined">
      <ModelViewer
        :key="rendererType"
        :segbinUrl="segbinUrl"
        :rendererType="rendererType"
        :roughness="roughness"
        :metalness="metalness"
        :envIntensity="envIntensity"
        :specularStrength="specularStrength"
        :ambientStrength="ambientStrength"
        :baseColorTint="baseColorTint"
        :envMapUrl="envMapUrl"
        :ssaoEnabled="ssaoEnabled"
        :debugPreview="rendererType === 'webgpu' ? debugPreview : 'none'"
        @model-loaded="onModelLoaded"
      />
      <div v-show="!segbinUrl" class="placeholder">
        Upload a 3D model to see the interactive preview
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onBeforeUnmount } from 'vue';
import ModelViewer from './components/ModelViewer.vue';
import SliderControl from './components/SliderControl.vue';

const loading = ref(false);
const error = ref('');
const segbinUrl = ref<string | null>(null);
const timing = ref<{ api?: number; model?: number; slice?: number; parse?: number; ray?: number; segBvh?: number; gap?: number; arc?: number; total?: number }>({});

const roughness = ref(0.10);
const metalness = ref(0.0);
const envIntensity = ref(1.0);
const specularStrength = ref(1.0);
const ambientStrength = ref(0.5);
const baseColorTint = ref('#e8e0d4');
const ssaoEnabled = ref(true);
const envMapUrl = ref('ferndale_studio_07_1k.hdr');
const envMapFiles = [
  'ferndale_studio_07_1k.hdr',
  'ferndale_studio_12_1k.hdr',
  'horn-koppe_spring_1k.hdr',
  'photo_studio_01_1k.hdr',
  'wooden_studio_03_1k.hdr',
];

// Feature-detect WebGPU
let webgpuAvailable = false;
let webgpuReason = '';
if (typeof navigator === 'undefined') {
  webgpuReason = 'no navigator (SSR?)';
} else if (typeof (navigator as any).gpu === 'undefined') {
  webgpuReason = 'navigator.gpu not found — browser/OS may not support WebGPU, or it needs a flag';
} else {
  webgpuAvailable = true;
}
const rendererType = ref<string>(webgpuAvailable ? 'webgpu' : 'webgl2');
const debugPreview = ref<string>('none');
log(`Renderer: ${rendererType.value}${webgpuReason ? ' (' + webgpuReason + ')' : ''}`);

function fmt(ms: number | undefined): string {
  if (ms === undefined || ms === null) return '-';
  if (ms < 1) return '<1ms';
  return ms + 'ms';
}

function pct(ms: number | undefined): string {
  if (ms === undefined || ms === null) return '';
  const total = timing.value?.total;
  if (!total || total <= 0) return '';
  return ((ms / total) * 100).toFixed(1) + '%';
}

function log(msg: string) {
  console.log(`[App] ${msg}`);
}

const sidebarWidth = ref(320);
const viewerPercent = ref(60);
const resizing = ref(false);
let resizeStartVal = 0;
let resizeStartPercent = 60;

function getResizeClient(e: MouseEvent | TouchEvent): number {
  return 'touches' in e ? e.touches[0] : e;
}

function isMobile() {
  return window.innerWidth <= 768;
}

function startResize(e: MouseEvent | TouchEvent) {
  resizing.value = true;
  const pt = getResizeClient(e);
  if (isMobile()) {
    resizeStartVal = pt.clientY;
    resizeStartPercent = viewerPercent.value;
  } else {
    resizeStartVal = pt.clientX;
  }
  document.addEventListener('mousemove', onResize);
  document.addEventListener('mouseup', stopResize);
  document.addEventListener('touchmove', onResize, { passive: false });
  document.addEventListener('touchend', stopResize);
  e.preventDefault();
}

function onResize(e: MouseEvent | TouchEvent) {
  if (!resizing.value) return;
  if (isMobile()) {
    const pt = getResizeClient(e);
    const dy = pt.clientY - resizeStartVal;
    viewerPercent.value = Math.max(40, Math.min(85, resizeStartPercent - (dy / window.innerHeight) * 100));
  } else {
    const pt = getResizeClient(e);
    sidebarWidth.value = Math.max(220, Math.min(600, pt.clientX));
  }
}

function stopResize() {
  resizing.value = false;
  document.removeEventListener('mousemove', onResize);
  document.removeEventListener('mouseup', stopResize);
  document.removeEventListener('touchmove', onResize);
  document.removeEventListener('touchend', stopResize);
}

onBeforeUnmount(() => {
  document.removeEventListener('mousemove', onResize);
  document.removeEventListener('mouseup', stopResize);
  document.removeEventListener('touchmove', onResize);
  document.removeEventListener('touchend', stopResize);
});

// Load segbin from ?segbin= query param (for headless/playwright mode)
const segbinParam = new URLSearchParams(location.search).get('segbin');
if (segbinParam) {
  segbinUrl.value = segbinParam;
}

function onModelLoaded(ms: number) {
  timing.value.model = ms;
  log(`Model loaded: ${ms}ms`);
}

async function loadBuiltin(model: string) {
  loading.value = true;
  error.value = '';
  timing.value = {};
  const t0 = performance.now();

  try {
    segbinUrl.value = null;
    const response = await fetch('/api/preview/builtin', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model }),
    });
    if (!response.ok) {
      const data = await response.json();
      throw new Error(data.error || 'Pipeline failed');
    }
    const data = await response.json();
    timing.value.api = Math.round(performance.now() - t0);
    if (data.timing) Object.assign(timing.value, data.timing);
    log(`API round-trip: ${timing.value.api}ms`);
    segbinUrl.value = data.segbin || null;
    log(`[loadBuiltin] segbinUrl='${segbinUrl.value}'`);
  } catch (e: any) {
    error.value = e.message;
  } finally {
    loading.value = false;
  }
}

const uploadModel = async (event: Event) => {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (!file) return;

  loading.value = true;
  error.value = '';
  timing.value = {};
  const formData = new FormData();
  formData.append('model', file);

  const t0 = performance.now();
  try {
    const response = await fetch('/api/preview', {
      method: 'POST',
      body: formData,
    });
    if (!response.ok) {
      const data = await response.json();
      throw new Error(data.error || 'Pipeline failed');
    }
    const data = await response.json();
    timing.value.api = Math.round(performance.now() - t0);
    if (data.timing) Object.assign(timing.value, data.timing);
    log(`API round-trip: ${timing.value.api}ms`);
    segbinUrl.value = data.segbin || null;
    log(`[uploadModel] segbinUrl='${segbinUrl.value}'`);
  } catch (e: any) {
    error.value = e.message;
  } finally {
    loading.value = false;
  }
};
</script>

<style>
/* ===== CSS Custom Properties (Light Mode Default) ===== */
:root {
  --app-bg: #fafafa;
  --app-text: #333;
  --app-text-secondary: #555;
  --app-text-muted: #999;
  --app-border: #ddd;
  --app-border-light: #ccc;
  --app-input-bg: #fff;
  --app-input-text: #333;
  --app-status-bg: #e3f2fd;
  --app-status-text: #1565c0;
  --app-status-error-bg: #ffebee;
  --app-status-error-text: #c62828;
  --app-input-bg: #fff;
  --app-input-text: #333;

  color-scheme: light dark;
  --app-slider-fill: #4a90d9;
  --app-slider-thumb: #4a90d9;
}

@media (prefers-color-scheme: dark) {
  :root {
    --app-bg: #1a1a2e;
    --app-text: #e0e0e0;
    --app-text-secondary: #b0b0b0;
    --app-text-muted: #6b6b80;
    --app-border: #2e2e42;
    --app-border-light: #3a3a52;
    --app-input-bg: #232337;
    --app-input-text: #e0e0e0;
    --app-status-bg: #1a2744;
    --app-status-text: #90caf9;
    --app-status-error-bg: #3e1a1a;
    --app-status-error-text: #ef9a9a;
    --app-input-bg: #232337;
    --app-input-text: #e0e0e0;
  }
}

* { margin: 0; padding: 0; box-sizing: border-box; }
html, body, #app { height: 100%; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: var(--app-bg); color: var(--app-text); }

.app { display: flex; height: 100vh; }

.sidebar {
  min-width: 220px; padding: 20px; border-right: 1px solid var(--app-border);
  background: var(--app-bg); overflow-y: auto; flex-shrink: 0;
}
.sidebar h3 { margin-bottom: 16px; color: var(--app-text); }

.builtin-buttons { display: flex; gap: 6px; margin-bottom: 8px; }
.builtin-buttons button {
  flex: 1; padding: 6px 10px; border: 1px solid var(--app-border-light);
  border-radius: 4px; background: var(--app-input-bg); color: var(--app-input-text);
  cursor: pointer; font-size: 13px;
}
.builtin-buttons button:hover:not(:disabled) { filter: brightness(1.2); }
.builtin-buttons button:disabled { opacity: 0.5; cursor: default; }

.builtin-divider { margin: 8px 0 6px; font-size: 12px; color: var(--app-text-muted); }
.sidebar h4 { margin: 10px 0 4px; color: var(--app-text-secondary); font-size: 13px; }

.sidebar input[type="file"] {
  width: 100%; padding: 8px 10px; border: 1px solid var(--app-border-light);
  border-radius: 4px; font-size: 13px; background: var(--app-input-bg);
  color: var(--app-input-text); cursor: pointer;
}

#material-controls label {
  display: block; margin-top: 4px; color: var(--app-text-secondary);
  font-size: 12px;
}
#material-controls label span {
  float: right; color: var(--app-text-muted);
}
#material-controls input[type="range"]:not(.slider-input) {
  width: 100%; margin: 1px 0 2px; height: 14px;
}
#material-controls .color-picker {
  width: 100%; height: 26px; padding: 1px; border: 1px solid var(--app-border-light);
  border-radius: 3px; background: var(--app-input-bg); cursor: pointer;
  box-sizing: border-box;
}
.renderer-select {
  width: 100%; padding: 5px 6px; margin-bottom: 4px;
  border: 1px solid var(--app-border-light); border-radius: 4px;
  background: var(--app-input-bg); color: var(--app-input-text);
  font-size: 13px; cursor: pointer;
}
.renderer-select option:disabled {
  color: var(--app-text-muted);
}

.status { margin-top: 12px; padding: 8px 12px; background: var(--app-status-bg); border-radius: 4px; color: var(--app-status-text); font-size: 14px; }
.status.error { background: var(--app-status-error-bg); color: var(--app-status-error-text); }
.timing-table { font-size: 12px; opacity: 0.8; margin-top: 4px; width: 100%; border-collapse: collapse; }
.timing-table td { padding: 1px 4px; }
.timing-table td:first-child { padding-left: 0; }
.t-ms { text-align: right; font-variant-numeric: tabular-nums; width: 60px; }
.t-pct { text-align: right; width: 48px; color: var(--app-text-muted); }

.resize-handle { width: 4px; cursor: col-resize; background: var(--app-border); flex-shrink: 0; }
.resize-handle:hover { background: var(--app-text-muted); }

.viewer { flex: 1; position: relative; min-width: 0; background: var(--app-bg); overscroll-behavior: none; touch-action: none; overflow: hidden; }
.placeholder { display: flex; align-items: center; justify-content: center; height: 100%; color: var(--app-text-muted); font-size: 18px; padding: 20px; text-align: center; }

@media (max-width: 768px) {
  .app { flex-direction: column; }
  .sidebar { width: 100% !important; flex: 1; overflow-y: auto; border-right: none; border-top: 1px solid var(--app-border); padding: 12px; }
  .sidebar h3 { margin-bottom: 10px; }
  .viewer { flex: none; }
  .resize-handle { width: 100%; height: 12px; cursor: row-resize; position: relative; display: flex; align-items: center; justify-content: center; background: transparent; }
  .resize-handle::after { content: ''; display: block; width: 40px; height: 4px; border-radius: 2px; background: var(--app-border); }
  #material-controls { display: flex; flex-wrap: wrap; gap: 4px; }
  #material-controls h4 { width: 100%; margin-top: 0; }
  #material-controls > label:not(.checkbox-label) { flex: 0 0 48%; }
  #material-controls > input[type="range"] { flex: 0 0 48%; }
  .builtin-buttons { flex-wrap: wrap; }
  .status { margin-top: 6px; }
}
</style>
