<template>
  <div class="slider-control">
    <div class="slider-track-wrap">
      <!-- Fill background: colored up to %, gray after -->
      <div class="slider-bg">
        <div class="slider-fill" :style="{ width: fillPercent + '%' }"></div>
      </div>
      <!-- Label & value overlaid -->
      <span class="slider-label">{{ label }}</span>
      <span class="slider-value">{{ displayValue }}</span>
      <!-- Invisible range input for interaction -->
      <input
        type="range"
        class="slider-input"
        :min="min"
        :max="max"
        :step="step"
        :value="modelValue"
        @input="$emit('update:modelValue', parseFloat(($event.target as HTMLInputElement).value))"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';

const props = withDefaults(defineProps<{
  label: string;
  modelValue: number;
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
  decimals?: number;
}>(), {
  min: 0,
  max: 1,
  step: 0.01,
  unit: '',
  decimals: 2,
});

defineEmits<{ 'update:modelValue': [value: number] }>();

const fillPercent = computed(() => {
  return ((props.modelValue - props.min) / (props.max - props.min)) * 100;
});

const displayValue = computed(() => {
  const v = props.modelValue.toFixed(props.decimals);
  return props.unit ? `${v}${props.unit}` : v;
});
</script>

<style scoped>
.slider-control {
  margin: 2px 0;
}

.slider-track-wrap {
  position: relative;
  height: 26px;
  cursor: pointer;
}

/* Background bar (gray) with fill (colored) */
.slider-bg {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 26px;
  border-radius: 13px;
  background: var(--slider-bg-empty, var(--app-border));
  overflow: hidden;
  z-index: 1;
}

.slider-fill {
  height: 100%;
  border-radius: 13px;
  background: var(--slider-fill, #3399ff);
  transition: width 0.08s ease;
  z-index: 2;
}

/* Label and value overlaid on the bar */
.slider-label {
  position: absolute;
  top: 0;
  left: 12px;
  height: 26px;
  line-height: 26px;
  font-size: 12px;
  font-weight: 500;
  color: #fff;
  text-shadow: 0 0 6px rgba(0,0,0,0.7);
  pointer-events: none;
  z-index: 4;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 60%;
}

.slider-value {
  position: absolute;
  top: 0;
  right: 12px;
  height: 26px;
  line-height: 26px;
  font-size: 12px;
  font-weight: 500;
  color: #fff;
  text-shadow: 0 0 6px rgba(0,0,0,0.7);
  pointer-events: none;
  z-index: 4;
  font-variant-numeric: tabular-nums;
}

/* Invisible range input — covers the whole bar for click/drag */
.slider-input {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 26px;
  margin: 0;
  padding: 0;
  -webkit-appearance: none;
  appearance: none;
  background: transparent;
  z-index: 5;
  cursor: pointer;
  opacity: 0;
}

/* Hide the thumb and track */
.slider-input::-webkit-slider-runnable-track {
  height: 26px;
  background: transparent;
  border: none;
}

.slider-input::-moz-range-track {
  height: 26px;
  background: transparent;
  border: none;
}

.slider-input::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 1px;
  height: 26px;
  background: transparent;
  border: none;
}

.slider-input::-moz-range-thumb {
  width: 1px;
  height: 26px;
  background: transparent;
  border: none;
}

.slider-input:focus {
  outline: none;
}
</style>
