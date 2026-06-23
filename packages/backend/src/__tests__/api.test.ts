import { describe, it, expect, beforeAll, afterAll } from 'bun:test';
import { createApp, config as defaultConfig } from '../index';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import type { Server } from 'node:http';

type JsonResponse = Record<string, unknown>;

// Test directories
const TEST_UPLOAD_DIR = mkdtempSync(join(tmpdir(), 'print-preview-test-uploads-'));
const TEST_OUTPUT_DIR = mkdtempSync(join(tmpdir(), 'print-preview-test-output-'));

// Mock: scripts that parse the CLI args, create expected output files, then exit 0.
// The real pipeline expects orca-slicer to create GCode in --outputdir
const MOCK_ORCA = join(import.meta.dir, 'mock-orca-slicer.sh');

function createTestApp() {
  return createApp({
    uploadDir: TEST_UPLOAD_DIR,
    outputDir: TEST_OUTPUT_DIR,
    orcaSlicerBin: MOCK_ORCA,
  });
}

describe('API Endpoints', () => {
  let server: Server;
  let baseUrl: string;

  beforeAll(async () => {
    const app = createTestApp();
    await new Promise<void>((resolve) => {
      server = app.listen(0, () => {
        const addr = server.address();
        if (addr && typeof addr === 'object') {
          baseUrl = `http://localhost:${addr.port}`;
        }
        resolve();
      });
    });
  });

  afterAll(() => {
    server?.close();
    rmSync(TEST_UPLOAD_DIR, { recursive: true, force: true });
    rmSync(TEST_OUTPUT_DIR, { recursive: true, force: true });
  });

  // --- Health check ---

  it('GET /api/health returns ok status', async () => {
    const res = await fetch(`${baseUrl}/api/health`);
    expect(res.status).toBe(200);
    const body = await res.json() as JsonResponse;
    expect(body.status).toBe('ok');
    expect(body.timestamp).toBeNumber();
  });

  // --- Preview endpoint: validation ---

  it('POST /api/preview without file returns 400', async () => {
    const form = new FormData();
    const res = await fetch(`${baseUrl}/api/preview`, {
      method: 'POST',
      body: form,
    });
    expect(res.status).toBe(400);
    const body = await res.json() as JsonResponse;
    expect(body.error).toContain('No model file');
  });

  it('POST /api/preview with unsupported file type returns 400', async () => {
    const form = new FormData();
    const badFile = new File(['fake data'], 'test.txt', { type: 'text/plain' });
    form.append('model', badFile);
    const res = await fetch(`${baseUrl}/api/preview`, {
      method: 'POST',
      body: form,
    });
    expect(res.status).toBe(400);
    const body = await res.json() as JsonResponse;
    expect(body.error).toContain('Unsupported file type');
    expect(body.error).toContain('.txt');
  });

  it('POST /api/preview with .stl file returns 200 (mocked pipeline)', async () => {
    const form = new FormData();
    const stlContent = Buffer.alloc(84, 0); // minimal STL binary header
    const stlFile = new File([stlContent], 'test.stl', { type: 'application/octet-stream' });
    form.append('model', stlFile);

    const res = await fetch(`${baseUrl}/api/preview`, {
      method: 'POST',
      body: form,
    });
    expect(res.status).toBe(200);
    const body = await res.json() as JsonResponse;
    expect(body.id).toBeString();
    if (body.segbin) {
      expect(body.segbin).toMatch(/\/previews\/.+\/preview\.segbin$/);
    }
  });

  it('POST /api/preview with .obj file returns 200 (mocked pipeline)', async () => {
    const form = new FormData();
    const objContent = Buffer.from('v 0 0 0\n');
    const objFile = new File([objContent], 'test.obj', { type: 'application/octet-stream' });
    form.append('model', objFile);

    const res = await fetch(`${baseUrl}/api/preview`, {
      method: 'POST',
      body: form,
    });
    expect(res.status).toBe(200);
    const body = await res.json() as JsonResponse;
    expect(body.id).toBeString();
  });

  it('concurrent uploads produce unique IDs', async () => {
    const results = await Promise.all(
      Array.from({ length: 5 }, async () => {
        const form = new FormData();
        form.append('model', new File(['data'], `model.stl`));
        const res = await fetch(`${baseUrl}/api/preview`, {
          method: 'POST',
          body: form,
        });
        return (await res.json() as JsonResponse).id as string;
      })
    );
    // All IDs should be unique
    expect(new Set(results).size).toBe(5);
  });

  // --- Static file serving ---

  it('serves static files from /previews', async () => {
    // Write a test file
    const testFile = join(TEST_OUTPUT_DIR, 'test.txt');
    writeFileSync(testFile, 'hello world');

    const res = await fetch(`${baseUrl}/previews/test.txt`);
    expect(res.status).toBe(200);
    const text = await res.text();
    expect(text).toBe('hello world');
  });

  it('returns 404 for non-existent preview file', async () => {
    const res = await fetch(`${baseUrl}/previews/nonexistent.png`);
    expect(res.status).toBe(404);
  });
});
