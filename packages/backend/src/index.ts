import express from 'express';
import multer from 'multer';
import { execSync, spawnSync } from 'node:child_process';
import { mkdirSync, existsSync, renameSync, readFileSync, writeFileSync, statSync, readdirSync, realpathSync } from 'node:fs';
import { join, normalize } from 'node:path';
import { randomUUID } from 'node:crypto';

// --- Timestamped logging ---
function log(msg: string) {
  const now = new Date();
  const ts = `${String(now.getHours()).padStart(2,'0')}:${String(now.getMinutes()).padStart(2,'0')}:${String(now.getSeconds()).padStart(2,'0')}.${String(now.getMilliseconds()).padStart(3,'0')}`;
  console.log(`[${ts}] ${msg}`);
}
function warn(msg: string) { log(`⚠ ${msg}`); }


/** Return elapsed ms since a Date, formatted as a human-readable string. */
function elapsed(since: Date): string {
  const ms = Date.now() - since.getTime();
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(2)}s`;
  return `${(ms / 60_000).toFixed(2)}m`;
}

/** Return a human-friendly file size string. */
function fileSize(path: string): string {
  try {
    const bytes = statSync(path).size;
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  } catch { return '?'; }
}

// --- Shared pipeline: slice → segbin ---

interface PipelineParams {
  stlPath: string;
  modelName: string;
  workDir: string;
  outputDir: string;
  cfg: typeof config;
  logPrefix: string;
}

/** Convert Rust Duration Debug (e.g. "1s", "185ms", "902µs") → ms. */
function parseDur(s: string): number | undefined {
  const m = s.match(/^([\d.]+)(s|ms|µs)$/);
  if (!m) return undefined;
  const v = parseFloat(m[1]);
  if (m[2] === 's') return v * 1000;
  if (m[2] === 'ms') return v;
  return v * 0.001; // µs
}

/** Parse gcode-to-segbin stderr into structured timing. */
function parseG2sTiming(stderr: string): Record<string, number> {
  const t: Record<string, number> = {};
  const m1 = stderr.match(/Parsed .+ \([\d.]+ merged, ([\d.]+)ms\)/);
  if (m1) t.parse = parseFloat(m1[1]);

  const m2 = stderr.match(/Ray cull: (\d+) rays, .+ culled \(ray=([^,]+), seg_bvh=([^,]+), gap=([^)]+)\)/);
  if (m2) {
    t.rays = parseInt(m2[1]);
    const r = parseDur(m2[2]); if (r !== undefined) t.ray = r;
    const s = parseDur(m2[3]); if (s !== undefined) t.segBvh = s;
    const g = parseDur(m2[4]); if (g !== undefined) t.gap = g;
  }

  const m3 = stderr.match(/Arc subdivision: .+ \(([^)]+)\)/);
  if (m3) { const a = parseDur(m3[1]); if (a !== undefined) t.arc = a; }

  const m4 = stderr.match(/Total: ([\d.]+[a-zµ]+)/);
  if (m4) { const u = parseDur(m4[1]); if (u !== undefined) t.total = u; }

  return t;
}

async function runPipeline(p: PipelineParams): Promise<Record<string, number> | undefined> {
  const pipelineStart = new Date();
  const segbinPath = join(p.workDir, 'preview.segbin');


  log(`[${p.logPrefix}] Slicing ${p.modelName} (${fileSize(p.stlPath)})...`);
  const sliceStart = Date.now();
  const resDir = p.cfg.orcaResourcesDir;
  const machineSettings = `${resDir}/profiles/BBL/machine/Bambu Lab A1 mini 0.4 nozzle.json`;
  const processSettings = `${resDir}/profiles/BBL/process/0.20mm Standard @BBL A1M.json`;
  const filamentSettings = `${resDir}/profiles/BBL/filament/Bambu PLA Basic @BBL A1M.json`;

  let gcodeFound = false;
  let gcodePath = '';

  const sliceArgs = [
    `--slice`, `1`,
    `--orient`, `0`,
    `--load-settings`, `"${machineSettings};${processSettings}"`,
    `--load-filaments`, `"${filamentSettings}"`,
    `--outputdir`, p.workDir,
    p.stlPath,
  ];


  try {
    execSync(
      `${p.cfg.orcaSlicerBin} ${sliceArgs.join(' ')}`,
      { stdio: 'pipe', timeout: 120_000, env: { ...process.env, ORCA_KEEP_INTERNAL: '1' } }
    );

    const files = readdirSync(p.workDir);
    gcodePath = files.find((f: string) => f.endsWith('.gcode')) || '';
    gcodeFound = !!gcodePath;
    gcodePath = gcodePath ? join(p.workDir, gcodePath) : '';
  } catch (err: any) {
    warn(`[${p.logPrefix}] Slicing failed: ${err.message}`);
  }

  let g2sTiming: Record<string, number> | undefined;
  const sliceMs = Date.now() - sliceStart;
  if (gcodeFound && existsSync(gcodePath)) {
    log(`[${p.logPrefix}]   ↳ sliced in ${elapsed(new Date(sliceStart))}, GCode=${fileSize(gcodePath)}`);

    const g2sArgs = [gcodePath, segbinPath];
    if (existsSync(p.stlPath)) { g2sArgs.push(p.stlPath); g2sArgs.push('--cull-method'); g2sArgs.push('ray'); }
    const g2sResult = spawnSync(p.cfg.gcodeToSegbinBin, g2sArgs, {
      stdio: 'pipe', timeout: 120_000,
    });
    if (g2sResult.status === 0) {
      log(`[${p.logPrefix}]   ↳ converted in ${elapsed(new Date(sliceStart))}, segbin=${fileSize(segbinPath)}`);
      const stderr = g2sResult.stderr?.toString() || '';
      g2sTiming = parseG2sTiming(stderr);
    } else {
      warn(`[${p.logPrefix}] gcode-to-segbin failed: ${g2sResult.stderr?.toString() || g2sResult.stdout?.toString()}`);
    }
  }

  log(`[${p.logPrefix}] Done (total ${elapsed(pipelineStart)}).`);
  const result = g2sTiming || {};
  result.slice = sliceMs;
  // Combine slice + g2s total for correct % baseline
  const g2sTotal = result.total || 0;
  result.total = sliceMs + g2sTotal;
  return result;
}

// --- Configuration (overridable for testing) ---

/** Resolve the orca-slicer binary: env override → nix profile → system PATH. */
function resolveOrcaSlicerBin(): string {
  if (process.env.ORCA_SLICER_BIN) return process.env.ORCA_SLICER_BIN;
  if (existsSync('/home/widget/.nix-profile/bin/orca-slicer'))
    return '/home/widget/.nix-profile/bin/orca-slicer';
  return 'orca-slicer';
}

/**
 * Resolve the OrcaSlicer resources directory.
 *
 * Checks (in order):
 *   1. $ORCA_RESOURCES_DIR env var
 *   2. The Nix init script that wraps the extracted AppImage tree
 *      (globs /nix/store/*OrcaSlicer-init for a -w arg with profiles)
 *   3. Common relative paths from the resolved binary location
 *   4. Common system share paths
 */
function resolveOrcaResourcesDir(): string {
  if (process.env.ORCA_RESOURCES_DIR) return process.env.ORCA_RESOURCES_DIR;

  // ── Nix: find the extracted AppImage tree via the bwrap init script ──
  try {
    const nixInits = readdirSync('/nix/store').filter(
      (f: string) => f.endsWith('-OrcaSlicer-init'),
    );
    for (const init of nixInits) {
      const content = readFileSync(join('/nix/store', init), 'utf-8');
      const m = content.match(/-w\s+(\/nix\/store\/[^\s]+)/);
      if (m && existsSync(join(m[1], 'resources', 'profiles', 'BBL'))) {
        return join(m[1], 'resources');
      }
    }
  } catch { /* not on Nix or no init scripts */ }

  // ── Probe common real-binary-relative locations ──
  const binPath = resolveOrcaSlicerBin();
  let realBinDir: string;
  try {
    realBinDir = dirname(realpathSync(binPath));
  } catch {
    realBinDir = dirname(binPath);
  }

  const candidates = [
    // Bundled (AppImage extract layout, macOS .app, manual install)
    join(realBinDir, 'resources'),
    join(dirname(binPath), 'resources'),
    // System package: /usr/share/orca-slicer/resources
    join(dirname(binPath), '..', 'share', 'orca-slicer', 'resources'),
    join(realBinDir, '..', 'share', 'orca-slicer', 'resources'),
    // Source build: binary at build/src/orca-slicer, resources at repo root
    join(dirname(binPath), '..', '..', 'resources'),
    join(realBinDir, '..', '..', 'resources'),
  ];

  const seen = new Set<string>();
  for (const dir of candidates) {
    const norm = normalize(dir);
    if (seen.has(norm)) continue;
    seen.add(norm);
    if (existsSync(join(norm, 'profiles', 'BBL'))) return norm;
  }

  return '';
}

const repoRoot = join(import.meta.dir, '..', '..', '..');

export const config = {
  port: parseInt(process.env.PORT || '3000'),
  uploadDir: process.env.UPLOAD_DIR || '/tmp/print-preview-uploads',
  outputDir: process.env.OUTPUT_DIR || '/tmp/print-preview-output',
  orcaSlicerBin: resolveOrcaSlicerBin(),
  orcaResourcesDir: resolveOrcaResourcesDir(),
  builtinModelsDir: join(repoRoot, 'test_files'),
  gcodeToSegbinBin: process.env.GCODE_TO_SEGBIN_BIN
    || join(repoRoot, 'packages', 'gcode-to-segbin', 'target', 'release', 'gcode-to-segbin'),
};

const builtinModels: Record<string, string> = {
  'benchy': '3dbenchy.stl',
  'calibration-cube': 'calibration-cube.stl',
};

// --- Filter OBJ by object names (o tags from OrcaSlicer exporter) ---
export function createApp(overrides?: Partial<typeof config>) {
  const cfg = { ...config, ...overrides };

  // Ensure directories exist
  if (!existsSync(cfg.uploadDir)) mkdirSync(cfg.uploadDir, { recursive: true });
  if (!existsSync(cfg.outputDir)) mkdirSync(cfg.outputDir, { recursive: true });

  const upload = multer({ dest: cfg.uploadDir });
  const app = express();

  // Enable SharedArrayBuffer for WASM-based 3D viewer
  app.use((_req, res, next) => {
    res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
    res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
    next();
  });

  // Serve generated previews as static files
  app.use('/previews', express.static(cfg.outputDir));

  // Health check
  app.get('/api/health', (_req, res) => {
    res.json({ status: 'ok', timestamp: Date.now() });
  });

  // Main preview endpoint
  app.post('/api/preview', upload.single('model'), async (req, res) => {
    const file = req.file;
    if (!file) {
      res.status(400).json({ error: 'No model file uploaded' });
      return;
    }

    // Validate file type
    const ext = file.originalname?.toLowerCase().split('.').pop();
    if (!ext || !['stl', 'obj'].includes(ext)) {
      res.status(400).json({ error: `Unsupported file type: .${ext || 'unknown'}. Only .stl and .obj are supported.` });
      return;
    }

    const modelId = randomUUID();
    const workDir = join(cfg.outputDir, modelId);
    mkdirSync(workDir, { recursive: true });

    const stlPath = file.path;

    // Ensure uploaded file has .stl extension (multer strips it)
    const stlWithExt = stlPath + (ext ? '.' + ext : '.stl');
    if (stlWithExt !== stlPath) {
      renameSync(stlPath, stlWithExt);
    }

    try {
      const timing = await runPipeline({
        stlPath: stlWithExt,
        modelName: file.originalname,
        workDir,
        outputDir: cfg.outputDir,
        cfg,
        logPrefix: modelId,
      });

      res.json({
        id: modelId,
        segbin: existsSync(join(workDir, 'preview.segbin'))
          ? `/previews/${modelId}/preview.segbin`
          : undefined,
        timing,
      });
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : String(error);
      warn(`[${modelId}] Pipeline failed: ${message}`);
      res.status(500).json({ error: message });
    }
  });

  // Built-in model endpoint — loads from repo test_files/
  app.post('/api/preview/builtin', express.json(), async (req, res) => {
    const modelName = req.body?.model as string | undefined;
    if (!modelName || !builtinModels[modelName]) {
      res.status(400).json({
        error: `Unknown model "${modelName}". Available: ${Object.keys(builtinModels).join(', ')}`,
      });
      return;
    }

    const stlPath = join(cfg.builtinModelsDir, builtinModels[modelName]);
    if (!existsSync(stlPath)) {
      res.status(404).json({ error: `Builtin model not found: ${builtinModels[modelName]}` });
      return;
    }

    const modelId = randomUUID();
    const workDir = join(cfg.outputDir, modelId);
    mkdirSync(workDir, { recursive: true });

    try {
      const timing = await runPipeline({
        stlPath,
        modelName: builtinModels[modelName],
        workDir,
        outputDir: cfg.outputDir,
        cfg,
        logPrefix: modelId,
      });

      res.json({
        id: modelId,
        segbin: existsSync(join(workDir, 'preview.segbin'))
          ? `/previews/${modelId}/preview.segbin`
          : undefined,
        timing,
      });
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : String(error);
      warn(`[${modelId}] Pipeline failed: ${message}`);
      res.status(500).json({ error: message });
    }
  });

  // Explicit 404 for /previews paths (before SPA fallback catches them)
  app.use('/previews', (_req, res) => {
    res.status(404).json({ error: 'Preview file not found' });
  });

  // In dev mode, proxy non-API requests to Vite dev server for HMR.
  // Otherwise, serve the built frontend.
  const isDev = process.env.DEV === 'true' || process.env.NODE_ENV === 'development';
  const viteDevServer = process.env.VITE_DEV_SERVER || 'http://localhost:5173';

  if (isDev) {
    log(`🔁 Proxying non-API requests → ${viteDevServer}`);
    app.use(async (req, res, next) => {
      // Only proxy GET/HEAD requests that aren't API or previews
      if (req.path.startsWith('/api') || req.path.startsWith('/previews')) {
        return next();
      }

      try {
        const upstream = await fetch(`${viteDevServer}${req.originalUrl}`, {
          method: req.method,
          headers: new Headers(req.headers as Record<string, string>),
        });

        res.status(upstream.status);
        upstream.headers.forEach((value, key) => {
          // Don't forward hop-by-hop headers
          if (!['transfer-encoding', 'connection'].includes(key.toLowerCase())) {
            res.setHeader(key, value);
          }
        });

        if (upstream.body) {
          const buf = await upstream.arrayBuffer();
          res.send(Buffer.from(buf));
        } else {
          res.end();
        }
      } catch {
        // Vite not running — fall through to next handler
        next();
      }
    });
  }

  // Always try to serve built frontend as fallback (or primary in non-dev)
  const frontendDist = join(import.meta.dir, '..', '..', 'frontend', 'dist');
  if (existsSync(frontendDist)) {
    app.use(express.static(frontendDist));
    // SPA fallback: serve index.html for all non-API, non-static routes
    app.use((_req, res) => {
      res.sendFile(join(frontendDist, 'index.html'));
    });
  }

  return app;
}

// --- Default app instance ---
export const app = createApp();

// Only start the server if this file is run directly (not imported by tests)
if (import.meta.main) {
  const server = app.listen(config.port, () => {
    log(`🦷 Print Preview Backend running on http://localhost:${config.port}`);
    log(`   Uploads: ${config.uploadDir}`);
    log(`   Output:  ${config.outputDir}`);
  });
  server.on('error', (err: NodeJS.ErrnoException) => {
    if (err.code === 'EADDRINUSE') {
      console.error(`❌ Port ${config.port} is already in use. Kill the existing process or set PORT env var.`);
    } else {
      console.error('❌ Server error:', err.message);
    }
    process.exit(1);
  });
  process.on('SIGTERM', () => server.close());
  process.on('SIGINT', () => server.close());
}
