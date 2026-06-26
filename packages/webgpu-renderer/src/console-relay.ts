/**
 * Console relay — forwards browser console messages to the Vite dev server
 * terminal over the HMR WebSocket. Only activates in Vite dev mode when the
 * URL contains `?relay` or when running on a non-localhost host.
 *
 * Usage:
 *   import './console-relay'
 *
 * Then on the dev server terminal you'll see [📱] prefixed log lines.
 */

// Use a Symbol so we can detect our own wrapped functions
const kRelayed = Symbol('console-relay')

function startRelay(tag?: string) {
  if (!import.meta.hot) return // only in Vite dev mode
  if ((console as any)[kRelayed]) return // already installed

  // Safe JSON serialization for console arguments
  function serializeArg(arg: unknown): unknown {
    if (arg === null || arg === undefined) return String(arg)
    if (typeof arg === 'string' || typeof arg === 'number' || typeof arg === 'boolean') return arg

    if (arg instanceof Error) {
      return `${arg.name}: ${arg.message}\n${arg.stack?.split('\n').slice(0, 4).join('\n') ?? ''}`
    }

    try {
      const json = JSON.stringify(arg, (_key, val) => {
        if (typeof val === 'bigint') return val.toString()
        if (val instanceof Error) return `${val.name}: ${val.message}`
        if (val instanceof Uint8Array || val instanceof Float32Array) return `<${val.constructor.name} ${val.length}>`
        if (val instanceof ArrayBuffer) return `<ArrayBuffer ${val.byteLength}>`
        return val
      }, 2)
      // Keep it short for terminal readability
      if (json.length > 500) return json.slice(0, 500) + '…'
      return json
    } catch {
      return String(arg)
    }
  }

  function relay(level: string, ...args: unknown[]) {
    const serialized = args.map(serializeArg)
    try {
      import.meta.hot!.send('console-relay', { level, args: serialized, tag })
    } catch {
      // Ignore send failures (server disconnected, etc.)
    }
  }

  const origLog = console.log.bind(console)
  const origWarn = console.warn.bind(console)
  const origError = console.error.bind(console)
  const origInfo = console.info.bind(console)

  // We still call the original so it shows in the phone's console too
  console.log = (...args: unknown[]) => { origLog(...args); relay('log', ...args) }
  console.warn = (...args: unknown[]) => { origWarn(...args); relay('warn', ...args) }
  console.error = (...args: unknown[]) => { origError(...args); relay('error', ...args) }
  console.info = (...args: unknown[]) => { origInfo(...args); relay('info', ...args) }

  // Catch unhandled errors & promise rejections (not covered by console.error override)
  if (typeof window !== 'undefined') {
    window.onerror = (msg, source, line, col, err) => {
      const text = err?.stack ? `${err.stack}` : `${msg} (${source}:${line}:${col})`
      relay('error', '[UNCAUGHT]', text)
      // Don't prevent default — let browser devtools also see it
      return false
    }
    window.addEventListener('unhandledrejection', (e: PromiseRejectionEvent) => {
      relay('error', '[UNHANDLED REJECTION]', e.reason?.stack ?? String(e.reason))
    })
  }

  ;(console as any)[kRelayed] = true

  origLog(`[console-relay] Active${tag ? ` (tag: ${tag})` : ''} — forwarding console to dev server`)
}

// Auto-start when URL has ?relay or when on a remote host (not localhost)
const isRemote = typeof location !== 'undefined' &&
  location.hostname !== 'localhost' &&
  location.hostname !== '127.0.0.1' &&
  !location.hostname.endsWith('.local')

const hasRelayFlag = typeof location !== 'undefined' &&
  new URLSearchParams(location.search).has('relay')

if (isRemote || hasRelayFlag) {
  const tag = new URLSearchParams(location.search).get('tag') ?? undefined
  startRelay(tag)
}

export { startRelay }
