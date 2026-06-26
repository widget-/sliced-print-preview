/**
 * Vite plugin: relay browser console messages to the dev server terminal
 * via Vite's built-in HMR WebSocket. The client module (console-relay.ts)
 * sends events through import.meta.hot.send('console-relay', ...).
 *
 * Usage:
 *   import consoleRelay from './vite.plugin.console-relay'
 *   plugins: [consoleRelay()]
 */
import type { Plugin, ViteDevServer } from 'vite'

export default function consoleRelay(): Plugin {
  return {
    name: 'console-relay',
    configureServer(server: ViteDevServer) {
      server.ws.on('connection', () => {
        // Nothing needed per-connection — we listen on the global event
      })

      // Listen for custom console events from the browser
      server.ws.on('console-relay', (data: any) => {
        if (!data || !data.level) return
        const ts = new Date().toLocaleTimeString()
        const prefix = `[📱${data.tag ? ` ${data.tag}` : ''} ${ts}]`
        const args = data.args ?? []

        switch (data.level) {
          case 'error':
            console.error(prefix, ...args)
            break
          case 'warn':
            console.warn(prefix, ...args)
            break
          case 'info':
            console.info(prefix, ...args)
            break
          default:
            console.log(prefix, ...args)
        }
      })

      console.log('[console-relay] Listening for browser console messages over Vite HMR WebSocket')
    },
  }
}
