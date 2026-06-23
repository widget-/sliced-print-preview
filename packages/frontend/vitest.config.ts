/// <reference types="vitest/config" />
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { resolve } from 'path'

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@sliced/shared': resolve(__dirname, '../shared/src'),
      '@sliced/webgpu-renderer': resolve(__dirname, '../webgpu-renderer/src'),
    },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    include: ['src/**/*.{test,spec}.{ts,js}'],
    setupFiles: ['./vitest.setup.ts'],
  },
})
