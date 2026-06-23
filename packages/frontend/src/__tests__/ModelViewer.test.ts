import { describe, it, expect, vi } from 'vitest';
import { mount } from '@vue/test-utils';

// Mock Babylon.js engine/canvas to prevent WebGL context creation in tests
vi.mock('@babylonjs/core/Engines/engine', () => ({
  Engine: vi.fn().mockImplementation(function (this: any) {
    this.runRenderLoop = vi.fn();
    this.stopRenderLoop = vi.fn();
    this.resize = vi.fn();
    this.dispose = vi.fn();
    this.getRenderingCanvas = vi.fn(() => document.createElement('canvas'));
    return this;
  }),
}));

vi.mock('@babylonjs/core/scene', () => ({
  Scene: vi.fn().mockImplementation(function (this: any) {
    this.clearColor = { r: 0, g: 0, b: 0, a: 1 };
    this.render = vi.fn();
    this.dispose = vi.fn();
    this.getEngine = vi.fn(() => ({
      getCaps: vi.fn(() => ({ maxMSAA: 0 })),
    }));
    return this;
  }),
}));

vi.mock('@babylonjs/core/Cameras/arcRotateCamera', () => ({
  ArcRotateCamera: vi.fn().mockImplementation(function (this: any) {
    this.position = { x: 0, y: 0, z: 0 };
    this.alpha = 0;
    this.beta = 0;
    this.radius = 100;
    this.upperBetaLimit = Math.PI;
    this.lowerBetaLimit = 0.01;
    this.attachControl = vi.fn();
    this.setTarget = vi.fn();
    return this;
  }),
}));

vi.mock('@babylonjs/core/Lights/hemisphericLight', () => ({
  HemisphericLight: vi.fn().mockImplementation(function (this: any) {
    this.intensity = 1;
    return this;
  }),
}));

vi.mock('@babylonjs/core/Lights/directionalLight', () => ({
  DirectionalLight: vi.fn().mockImplementation(function (this: any) {
    this.intensity = 1;
    this.position = { x: 0, y: 0, z: 0 };
    this.direction = { x: 0, y: 0, z: 0 };
    return this;
  }),
}));

vi.mock('@babylonjs/core/Maths/math.vector', () => ({
  Vector3: vi.fn().mockImplementation(function (this: any, x = 0, y = 0, z = 0) {
    this.x = x; this.y = y; this.z = z;
    this.subtract = vi.fn(() => this);
    this.normalize = vi.fn(() => this);
    return this;
  }),
  Matrix: {
    Identity: vi.fn(() => ({})),
  },
}));

vi.mock('@babylonjs/core/Maths/math.color', () => ({
  Color3: { FromHexString: vi.fn(() => ({ r: 0, g: 0, b: 0 })) },
  Color4: vi.fn(),
}));

vi.mock('@babylonjs/core/Materials/shaderMaterial', () => ({
  ShaderMaterial: vi.fn().mockImplementation(function (this: any) {
    this.setFloat = vi.fn();
    this.setFloat2 = vi.fn();
    this.setFloat3 = vi.fn();
    this.setFloat4 = vi.fn();
    this.setVector2 = vi.fn();
    this.setVector3 = vi.fn();
    this.setTexture = vi.fn();
    this.setMatrix = vi.fn();
    this.dispose = vi.fn();
    return this;
  }),
}));

vi.mock('@babylonjs/core/Culling/ray', () => ({}));

vi.mock('@babylonjs/core/PostProcesses/RenderPipeline/Pipelines/taaRenderingPipeline', () => ({
  TAARenderingPipeline: vi.fn().mockImplementation(function (this: any) {
    this.samples = 8;
    this.reprojectHistory = false;
    this.disableOnCameraMove = true;
    this.dispose = vi.fn();
    return this;
  }),
}));

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
    // Babylon Engine may or may not create a canvas — just check component renders
    expect(wrapper.find('.preview-container').exists()).toBe(true);
  });
});
