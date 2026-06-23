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
        <label>Roughness <span>{{ roughness.toFixed(2) }}</span></label>
        <input type="range" min="0" max="1" step="0.01" v-model.number="roughness" />
        <label>Metalness <span>{{ metalness.toFixed(2) }}</span></label>
        <input type="range" min="0" max="1" step="0.01" v-model.number="metalness" />
        <label>Env Intensity <span>{{ envIntensity.toFixed(2) }}</span></label>
        <input type="range" min="0" max="2" step="0.01" v-model.number="envIntensity" />
        <label>Specular <span>{{ specularStrength.toFixed(2) }}</span></label>
        <input type="range" min="0" max="1" step="0.01" v-model.number="specularStrength" />
        <label>Ambient <span>{{ ambientStrength.toFixed(2) }}</span></label>
        <input type="range" min="0" max="2" step="0.01" v-model.number="ambientStrength" />
        <label>Color</label>
        <input type="color" v-model="baseColorTint" class="color-picker" />
      </div>
    </div>
    <div class="resize-handle" @mousedown="startResize" />
    <div class="viewer">
      <ModelViewer
        :segbinUrl="segbinUrl"
        :roughness="roughness"
        :metalness="metalness"
        :envIntensity="envIntensity"
        :specularStrength="specularStrength"
        :ambientStrength="ambientStrength"
        :baseColorTint="baseColorTint"
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

const loading = ref(false);
const error = ref('');
const segbinUrl = ref<string | null>(null);
const timing = ref<{ api?: number; model?: number; slice?: number; parse?: number; ray?: number; segBvh?: number; gap?: number; arc?: number; total?: number }>({});

const roughness = ref(0.10);
const metalness = ref(0.0);
const envIntensity = ref(0.25);
const specularStrength = ref(1.0);
const ambientStrength = ref(0.5);
const baseColorTint = ref('#e8e0d4');

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
const resizing = ref(false);

function startResize(e: MouseEvent) {
  resizing.value = true;
  document.addEventListener('mousemove', onResize);
  document.addEventListener('mouseup', stopResize);
  e.preventDefault();
}

function onResize(e: MouseEvent) {
  if (!resizing.value) return;
  sidebarWidth.value = Math.max(220, Math.min(600, e.clientX));
}

function stopResize() {
  resizing.value = false;
  document.removeEventListener('mousemove', onResize);
  document.removeEventListener('mouseup', stopResize);
}

onBeforeUnmount(() => {
  document.removeEventListener('mousemove', onResize);
  document.removeEventListener('mouseup', stopResize);
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
#material-controls input[type="range"] {
  width: 100%; margin: 1px 0 2px; height: 14px;
}
#material-controls .color-picker {
  width: 100%; height: 26px; padding: 1px; border: 1px solid var(--app-border-light);
  border-radius: 3px; background: var(--app-input-bg); cursor: pointer;
  box-sizing: border-box;
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

.viewer { flex: 1; position: relative; min-width: 0; background: var(--app-bg); }
.placeholder { display: flex; align-items: center; justify-content: center; height: 100%; color: var(--app-text-muted); font-size: 18px; padding: 20px; text-align: center; }
</style>
