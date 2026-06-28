// Vitest setup for jsdom environment
import { vi } from 'vitest';

// Mock ResizeObserver (not available in jsdom)
globalThis.ResizeObserver = vi.fn(function (this: any) {
  this.observe = vi.fn();
  this.unobserve = vi.fn();
  this.disconnect = vi.fn();
}) as any;
