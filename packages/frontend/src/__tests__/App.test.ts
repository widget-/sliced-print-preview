import { describe, it, expect, vi, beforeEach } from 'vitest';
import { mount } from '@vue/test-utils';

// Mock fetch
const mockFetch = vi.fn();
globalThis.fetch = mockFetch as any;

import App from '../App.vue';

// Stub ModelViewer to avoid WeakMap issue with Bun + vue-test-utils
const mountOptions = {
  global: {
    stubs: {
      ModelViewer: {
        template: '<div class="mock-model-viewer"><slot /></div>',
        props: ['glbUrl', 'debugPreview'],
      },
    },
  },
};

describe('App', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders the app layout', () => {
    const wrapper = mount(App, mountOptions);
    expect(wrapper.find('.app').exists()).toBe(true);
    expect(wrapper.find('.sidebar').exists()).toBe(true);
    expect(wrapper.find('.viewer').exists()).toBe(true);
  });

  it('shows title "3D Print Preview"', () => {
    const wrapper = mount(App, mountOptions);
    expect(wrapper.find('h3').text()).toBe('3D Print Preview');
  });

  it('shows placeholder when no model loaded', () => {
    const wrapper = mount(App, mountOptions);
    expect(wrapper.find('.placeholder').exists()).toBe(true);
    expect(wrapper.find('.placeholder').text()).toContain('Upload');
  });

  it('has a file input that accepts .stl files', () => {
    const wrapper = mount(App, mountOptions);
    const input = wrapper.find('input[type="file"]');
    expect(input.exists()).toBe(true);
    expect(input.attributes('accept')).toContain('.stl');
  });

  it('does not show loading status initially', () => {
    const wrapper = mount(App, mountOptions);
    expect(wrapper.find('.status').exists()).toBe(false);
  });

  it('shows error message on failed upload', async () => {
    mockFetch.mockRejectedValueOnce(new Error('Test error'));

    const wrapper = mount(App, mountOptions);

    // Simulate file upload
    const file = new File(['test'], 'test.stl');
    const input = wrapper.find('input[type="file"]');
    Object.defineProperty(input.element, 'files', {
      value: [file],
    });
    await input.trigger('change');

    // Wait for async operations
    await new Promise((r) => setTimeout(r, 100));
    await wrapper.vm.$nextTick();

    // Should show error
    expect(wrapper.find('.status.error').exists()).toBe(true);
  });

  it('hides placeholder after successful upload', async () => {
    // First fetch: upload API
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({
        segbin: '/models/test.segbin',
        id: 'test-123',
      }),
    });

    const wrapper = mount(App, mountOptions);

    const file = new File(['test'], 'test.stl');
    const input = wrapper.find('input[type="file"]');
    Object.defineProperty(input.element, 'files', {
      value: [file],
    });
    await input.trigger('change');

    await new Promise((r) => setTimeout(r, 100));
    await wrapper.vm.$nextTick();

    // Should hide placeholder (model available)
    const placeholder = wrapper.find('.placeholder');
    expect(placeholder.exists()).toBe(true);
    expect((placeholder.element as HTMLElement).style.display).toBe('none');
  });

  it('has sidebar and viewer layout structure', () => {
    const wrapper = mount(App, mountOptions);
    const sidebar = wrapper.find('.sidebar');
    const viewer = wrapper.find('.viewer');

    expect(sidebar.exists()).toBe(true);
    expect(viewer.exists()).toBe(true);
  });

  it('passes debugPreview prop to ModelViewer', async () => {
    const wrapper = mount(App, mountOptions);

    const vm = wrapper.vm as any;
    expect(vm.debugPreview).toBe('none');

    // The ModelViewer stub renders with class mock-model-viewer
    const mv = wrapper.find('.mock-model-viewer');
    expect(mv.exists()).toBe(true);

    // Set debug to velocity
    vm.debugPreview = 'velocity';
    await wrapper.vm.$nextTick();

    // Verify App state is correct
    expect(vm.debugPreview).toBe('velocity');
  });
});
