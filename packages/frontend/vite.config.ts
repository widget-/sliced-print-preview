import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import basicSsl from '@vitejs/plugin-basic-ssl'
import { resolve } from 'path'

export default defineConfig({
  plugins: [basicSsl(), vue()],
  resolve: {
    alias: {
      '@sliced/shared': resolve(__dirname, '../shared/src'),
      '@sliced/playcanvas-renderer': resolve(__dirname, '../playcanvas-renderer/src'),
    },
  },
  server: {
    host: '0.0.0.0',
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
