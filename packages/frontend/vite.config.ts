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
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
    proxy: {
      '/api': 'http://localhost:3000',
      '/previews': 'http://localhost:3000',
    },
  },
})
