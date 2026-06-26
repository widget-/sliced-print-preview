import { defineConfig } from 'vite'
import { resolve } from 'path'
import basicSsl from '@vitejs/plugin-basic-ssl'
import consoleRelay from './vite.plugin.console-relay'

export default defineConfig({
  plugins: [basicSsl(), consoleRelay()],
  resolve: {
    alias: {
      '@sliced/shared': resolve(__dirname, '../shared/src'),
    },
  },
  server: {
    https: true,
    host: '0.0.0.0',
  },
})
