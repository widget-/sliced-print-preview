import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import basicSsl from '@vitejs/plugin-basic-ssl'
import { resolve } from 'path'
import consoleRelay from '../webgpu-renderer/vite.plugin.console-relay'

export default defineConfig({
  plugins: [basicSsl(), vue(), consoleRelay()],
  resolve: {
    alias: {
      '@sliced/shared': resolve(__dirname, '../shared/src'),
      '@sliced/webgpu-renderer': resolve(__dirname, '../webgpu-renderer/src'),
      '@sliced/webgl-renderer': resolve(__dirname, '../webgl-renderer/src'),
    },
  },
  server: {
    host: '0.0.0.0',
    https: true,
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
