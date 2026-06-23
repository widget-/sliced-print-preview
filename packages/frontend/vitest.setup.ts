// Vitest setup: mock WebGL/Canvas APIs for jsdom environment
import { vi } from 'vitest';

// Mock HTMLCanvasElement.getContext to prevent WebGL errors
const mockContext = {
  getExtension: vi.fn().mockReturnValue(null),
  getParameter: vi.fn().mockReturnValue(null),
  createShader: vi.fn().mockReturnValue({}),
  shaderSource: vi.fn(),
  compileShader: vi.fn(),
  getShaderParameter: vi.fn().mockReturnValue(true),
  createProgram: vi.fn().mockReturnValue({}),
  attachShader: vi.fn(),
  linkProgram: vi.fn(),
  getProgramParameter: vi.fn().mockReturnValue(true),
  useProgram: vi.fn(),
  getAttribLocation: vi.fn().mockReturnValue(0),
  getUniformLocation: vi.fn().mockReturnValue({}),
  enableVertexAttribArray: vi.fn(),
  vertexAttribPointer: vi.fn(),
  uniformMatrix4fv: vi.fn(),
  uniform1i: vi.fn(),
  uniform1f: vi.fn(),
  uniform3fv: vi.fn(),
  uniform4fv: vi.fn(),
  createBuffer: vi.fn().mockReturnValue({}),
  bindBuffer: vi.fn(),
  bufferData: vi.fn(),
  createTexture: vi.fn().mockReturnValue({}),
  bindTexture: vi.fn(),
  texImage2D: vi.fn(),
  texParameteri: vi.fn(),
  activeTexture: vi.fn(),
  viewport: vi.fn(),
  clearColor: vi.fn(),
  clear: vi.fn(),
  enable: vi.fn(),
  disable: vi.fn(),
  blendFunc: vi.fn(),
  drawArrays: vi.fn(),
  drawElements: vi.fn(),
  getError: vi.fn().mockReturnValue(0),
  createFramebuffer: vi.fn().mockReturnValue({}),
  bindFramebuffer: vi.fn(),
  framebufferTexture2D: vi.fn(),
  checkFramebufferStatus: vi.fn().mockReturnValue(0),
};

HTMLCanvasElement.prototype.getContext = vi.fn((type: string) => {
  if (type === 'webgl2' || type === 'webgl' || type === 'experimental-webgl') {
    return mockContext as any;
  }
  return null;
}) as any;

// Mock ResizeObserver (not available in jsdom)
globalThis.ResizeObserver = vi.fn(function (this: any) {
  this.observe = vi.fn();
  this.unobserve = vi.fn();
  this.disconnect = vi.fn();
}) as any;

// Suppress console warnings during tests
const originalWarn = console.warn;
console.warn = (...args: any[]) => {
  if (typeof args[0] === 'string' && args[0].includes('Not implemented: HTMLCanvasElement')) return;
  if (typeof args[0] === 'string' && args[0].includes('Error creating WebGL context')) return;
  originalWarn(...args);
};
