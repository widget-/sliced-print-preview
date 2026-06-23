import { existsSync, readdirSync, readFileSync, realpathSync } from 'node:fs';
import { join, normalize, dirname } from 'node:path';

/** Resolve the orca-slicer binary: env override → nix profile → system PATH. */
export function resolveOrcaSlicerBin(): string {
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
export function resolveOrcaResourcesDir(): string {
  if (process.env.ORCA_RESOURCES_DIR) return process.env.ORCA_RESOURCES_DIR;

  // ── Nix: find the extracted AppImage tree via the bwrap init script ──
  try {
    const nixInits = readdirSync('/nix/store').filter(
      (f: string) => f.endsWith('-OrcaSlicer-init'),
    );
    for (const init of nixInits) {
      const content = readFileSync(join('/nix/store', init), 'utf-8');
      const m = content.match(/-w\s+(\/nix\/store\/[^\s]+)/);
      if (m && m[1] && existsSync(join(m[1], 'resources', 'profiles', 'BBL'))) {
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
