import { defineConfig, type Plugin } from 'vite'
import vue from '@vitejs/plugin-vue'
import basicSsl from '@vitejs/plugin-basic-ssl'
import { resolve } from 'path'

/** Prints console messages relayed from remote browsers (iPhone, etc.) to the dev terminal. */
function consoleRelay(): Plugin {
  return {
    name: 'console-relay',
    configureServer(server) {
      server.ws.on('console-relay', (data: { level: string; args: unknown[]; tag?: string }) => {
        const prefix = data.tag ? `[📱 ${data.tag}]` : '[📱]'
        const method = data.level === 'warn' ? console.warn
                    : data.level === 'error' ? console.error
                    : console.log
        method(prefix, ...data.args)
      })
    },
  }
}
export default defineConfig({
  plugins: [basicSsl(), vue(), consoleRelay()],
  resolve: {
    alias: {
      '@sliced/shared': resolve(__dirname, '../shared/src'),
      '@sliced/playcanvas-renderer': resolve(__dirname, '../playcanvas-renderer/src'),
      '@sliced/webgpu-renderer': resolve(__dirname, '../webgpu-renderer/src'),
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
