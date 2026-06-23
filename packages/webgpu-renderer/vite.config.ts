import { defineConfig } from 'vite'
import { resolve } from 'path'

export default defineConfig({
  resolve: {
    alias: {
      '@sliced/shared': resolve(__dirname, '../shared/src'),
    },
  },
})
