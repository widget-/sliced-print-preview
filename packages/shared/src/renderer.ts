export interface MaterialProps {
  roughness: number;
  metalness: number;
  envIntensity: number;
  specularStrength: number;
  ambientStrength: number;
  baseColorTint: string; // hex e.g. '#e8e0d4'
  ssaoIntensity?: number;
  ssaoRadius?: number;
  arcCurvature?: number;
  useRoleColors?: number;
}

export interface ScreenshotHooks {
  forceLOD(lod: number): void;
  getTriCount(): number;
}

export interface Renderer {
  /** Mount to a container that already contains a <canvas>. */
  mount(container: HTMLElement, canvas: HTMLCanvasElement): Promise<void>;
  /** Load a .segbin model. Returns total load time in ms. */
  loadModel(url: string): Promise<number>;
  /** Update material properties. */
  setMaterial(props: MaterialProps): void;
  /** Change the environment map. Re-initializes IBL textures. */
  setEnvMap(url: string): Promise<void>;
  /** Handle container resize. */
  resize(): void;
  /** Full teardown. */
  dispose(): void;
  /** Debug hooks for headless screenshot testing. */
  getScreenshotHooks(): ScreenshotHooks | null;
}
