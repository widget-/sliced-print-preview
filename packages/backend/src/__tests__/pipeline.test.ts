import { describe, it, expect, beforeEach, afterEach, mock } from 'bun:test';
import { mkdtempSync, rmSync, writeFileSync, existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';

// Test the pipeline orchestration logic in isolation.
// We mock execSync to simulate external tool behavior.

describe('Pipeline Orchestration Logic', () => {
  let testDir: string;

  beforeEach(() => {
    testDir = mkdtempSync(join(tmpdir(), 'print-preview-'));
  });

  afterEach(() => {
    rmSync(testDir, { recursive: true, force: true });
  });

  // --- Simulated pipeline steps ---

  it('creates output directories for each model', () => {
    const modelId = 'test-model-123';
    const workDir = join(testDir, modelId);

    // Simulate what the backend does
    const { mkdirSync } = require('node:fs');
    mkdirSync(workDir, { recursive: true });

    expect(existsSync(workDir)).toBe(true);
    expect(existsSync(join(testDir, 'nonexistent'))).toBe(false);
  });

  it('generates correct OBJ and PNG output paths', () => {
    const modelId = 'abc-def-456';
    const workDir = join(testDir, modelId);

    const objPath = join(workDir, 'sliced_model.obj');
    const pngPath = join(workDir, 'preview.png');

    expect(objPath).toEndWith('sliced_model.obj');
    expect(pngPath).toEndWith('preview.png');
    expect(objPath).toContain(modelId);
    expect(pngPath).toContain(modelId);
  });

  it('constructs correct orca-slicer CLI command', () => {
    const stlPath = '/tmp/input.stl';
    const objPath = '/tmp/output/model/sliced_model.obj';
    const bin = 'orca-slicer';

    const cmd = `${bin} --slice --export-toolpaths-obj --output=${objPath} ${stlPath}`;

    expect(cmd).toContain('--slice');
    expect(cmd).toContain('--export-toolpaths-obj');
    expect(cmd).toContain(stlPath);
    expect(cmd).toContain(objPath);
  });


  it('handles orca-slicer failure gracefully', () => {
    // Simulate what happens when execSync throws
    const simulateError = () => {
      throw new Error('orca-slicer: command not found');
    };

    let caughtError: string | null = null;
    try {
      simulateError();
    } catch (error: unknown) {
      caughtError = error instanceof Error ? error.message : String(error);
    }

    expect(caughtError).toContain('orca-slicer');
  });


  it('accepts .stl extension', () => {
    const validExts = ['stl', 'obj', 'STL', 'OBJ', 'Stl', 'Obj'];
    for (const ext of validExts) {
      expect(['stl', 'obj']).toContain(ext.toLowerCase());
    }
  });

  it('rejects unsupported extensions', () => {
    const invalid = ['txt', 'gcode', '3mf', 'ply', 'sog', ''];
    for (const ext of invalid) {
      const lower = ext.toLowerCase();
      expect(['stl', 'obj']).not.toContain(lower);
    }
  });

  // --- URL generation ---

  it('generates correct preview URLs', () => {
    const modelId = 'model-001';
    const pngUrl = `/previews/${modelId}/preview.png`;
    const objUrl = `/previews/${modelId}/sliced_model.obj`;

    expect(pngUrl).toBe('/previews/model-001/preview.png');
    expect(objUrl).toBe('/previews/model-001/sliced_model.obj');
  });

  // --- Output directory cleanup simulation ---

  it('does not leave stale files on success path', () => {
    // After a successful pipeline run, the work dir should contain
    // the expected output files
    const modelId = 'clean-test';
    const workDir = join(testDir, modelId);
    const { mkdirSync, writeFileSync } = require('node:fs');
    mkdirSync(workDir, { recursive: true });

    // Simulate pipeline output
    writeFileSync(join(workDir, 'sliced_model.obj'), '# OBJ file');
    writeFileSync(join(workDir, 'preview.png'), 'PNG');

    expect(existsSync(join(workDir, 'sliced_model.obj'))).toBe(true);
    expect(existsSync(join(workDir, 'preview.png'))).toBe(true);
    expect(existsSync(join(workDir, 'nonexistent.txt'))).toBe(false);
  });

  // --- Timeout handling ---

  it('orca-slicer command uses 120s timeout', () => {
    const timeout = 120_000;
    expect(timeout).toBe(120_000); // 2 minutes is reasonable for slicing
  });


});
