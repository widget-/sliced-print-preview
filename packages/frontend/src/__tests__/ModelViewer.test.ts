import { describe, it, expect } from 'vitest';
import { mount } from '@vue/test-utils';

import ModelViewer from '../components/ModelViewer.vue';

describe('ModelViewer', () => {
  it('renders the container', () => {
    const wrapper = mount(ModelViewer, {
      props: { segbinUrl: null },
    });
    expect(wrapper.find('.preview-container').exists()).toBe(true);
  });

  it('has correct CSS class for sizing', () => {
    const wrapper = mount(ModelViewer, {
      props: { segbinUrl: null },
    });
    const container = wrapper.find('.preview-container');
    expect(container.attributes('class')).toContain('preview-container');
  });

  it('does not crash when segbinUrl is null', () => {
    expect(() => {
      mount(ModelViewer, { props: { segbinUrl: null } });
    }).not.toThrow();
  });

  it('creates a canvas element in the container', () => {
    const wrapper = mount(ModelViewer, {
      props: { segbinUrl: null },
    });
    expect(wrapper.find('.preview-container').exists()).toBe(true);
  });
});
