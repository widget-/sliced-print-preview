import { generateBodyGeometry, generateCapGeometry } from './geometry';
import type { BodyGeometry, CapGeometry } from './geometry';
import { OrbitCamera } from './camera';
import { evaluateLOD } from './lod';
import type { SegmentTextures } from './buffer';
import type { SegbinData } from '@sliced/shared';
import { loadHDR } from './hdr';

// ── Shader imports (raw strings via Vite) ──
import bodyVertSrc from './shaders/body.vert?raw';
import capVertSrc from './shaders/cap.vert?raw';
import pbrFragSrc from './shaders/pbr.frag?raw';
import shadowVertSrc from './shaders/shadow.vert?raw';
import shadowFragSrc from './shaders/shadow.frag?raw';

export interface MaterialUniforms {
  roughness: number;
  metalness: number;
  envIntensity: number;
  specularStrength: number;
  ambientStrength: number;
  baseColorTint: [number, number, number];
  useRoleColors: number;
}

export class WebGLPipeline {
  gl: WebGL2RenderingContext;

  // ── Geometry (shared with WebGPU renderer) ──
  bodyGeos!: BodyGeometry[];
  capGeos!: CapGeometry[];
  bodyVBOs: WebGLBuffer[] = [];
  bodyIBOs: WebGLBuffer[] = [];
  bodyIC: number[] = [];
  capVBOs: WebGLBuffer[] = [];
  capIBOs: WebGLBuffer[] = [];
  capIC: number[] = [];

  // ── Programs ──
  bodyProgram!: WebGLProgram;
  capProgram!: WebGLProgram;
  shadowProgram!: WebGLProgram;

  // ── Uniform locations (body/cap share same layout) ──
  // Camera UBO
  cameraUBO!: WebGLBuffer;

  // Material UBO
  materialUBO!: WebGLBuffer;

  // Light UBO
  lightDirUBO!: WebGLBuffer;

  // Uniform locations
  u_lightDir!: WebGLUniformLocation;
  u_segTex!: WebGLUniformLocation;
  u_colorTex!: WebGLUniformLocation;
  u_texWidth!: WebGLUniformLocation;
  u_shadowTex!: WebGLUniformLocation;
  u_shadowVP!: WebGLUniformLocation;
  u_shadowSoftness!: WebGLUniformLocation;
  u_envTex!: WebGLUniformLocation;

  // Cap-specific
  u_cap_segTex!: WebGLUniformLocation;
  u_cap_colorTex!: WebGLUniformLocation;
  u_cap_texWidth!: WebGLUniformLocation;
  u_cap_shadowTex!: WebGLUniformLocation;
  u_cap_shadowVP!: WebGLUniformLocation;
  u_cap_shadowSoftness!: WebGLUniformLocation;
  u_cap_envTex!: WebGLUniformLocation;
  u_cap_lightDir!: WebGLUniformLocation;

  // Shadow program uniforms
  u_shadow_shadowVP!: WebGLUniformLocation;
  u_shadow_segTex!: WebGLUniformLocation;
  u_shadow_texWidth!: WebGLUniformLocation;

  // ── Shadow map ──
  shadowFBO!: WebGLFramebuffer;
  shadowTex!: WebGLTexture;
  shadowSize = 1024;

  // ── Environment map ──
  envTex: WebGLTexture | null = null;

  // ── Segment data ──
  segmentTextures!: SegmentTextures;
  segbinData!: SegbinData;

  // ── Instance buffers (per-LOD segment index arrays) ──
  lodInstanceBufs: WebGLBuffer[] = [];
  capInstanceBuf!: WebGLBuffer;

  // ── Material ──
  material: MaterialUniforms = {
    roughness: 0.65,
    metalness: 0,
    envIntensity: 1.0,
    specularStrength: 1,
    ambientStrength: 0.5,
    baseColorTint: [1, 0.878, 0.831],
    useRoleColors: 1,
  };
  lightDir: [number, number, number, number] = [0.416, -0.25, 0.872, 1];
  shadowSoftness = 2.0;

  // ── Model bounds for shadow VP ──
  _modelCenter: [number, number, number] = [0, 0, 0];
  _modelRadius = 100;

  constructor(gl: WebGL2RenderingContext) {
    this.gl = gl;
  }

  init() {
    const gl = this.gl;

    // ── Geometry (higher cross-section detail for WebGL since LOD is CPU-side) ──
    // LOD 0: 10 edge segments (44 ring verts), 5 interior rings — smooth curves up close
    // LOD 1: 6 edge segments (28 ring verts), 2 interior rings
    // LOD 2: 5 edge segments, 0 interior rings — flat distant quads
    const FLAT_W = 0.35;
    const HSCALE = 0.35;
    const BULGE_DEG = 40;
    const BULGE_RATIO = 0.5;
    this.bodyGeos = [
      generateBodyGeometry(FLAT_W, HSCALE, 10, 5, BULGE_DEG, BULGE_RATIO),  // LOD 0: 44 ring verts, 5 interior rings
      generateBodyGeometry(FLAT_W, HSCALE, 6, 2, BULGE_DEG, BULGE_RATIO),   // LOD 1: 28 ring verts, 2 interior rings
      generateBodyGeometry(FLAT_W, HSCALE, 3, 0, BULGE_DEG, BULGE_RATIO),   // LOD 2: 16 ring verts, 0 interior rings (culled early)
    ];
    this.capGeos = [
      generateCapGeometry(FLAT_W, HSCALE, 10, 6, BULGE_DEG, BULGE_RATIO),
      generateCapGeometry(FLAT_W, HSCALE, 6, 3, BULGE_DEG, BULGE_RATIO),
    ];

    for (let lod = 0; lod < 3; lod++) {
      const bg = this.bodyGeos[lod];
      const vb = gl.createBuffer()!;
      gl.bindBuffer(gl.ARRAY_BUFFER, vb);
      gl.bufferData(gl.ARRAY_BUFFER, bg.interleaved, gl.STATIC_DRAW);
      this.bodyVBOs.push(vb);

      const ib = gl.createBuffer()!;
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, bg.indices, gl.STATIC_DRAW);
      this.bodyIBOs.push(ib);
      this.bodyIC.push(bg.indices.length);
    }

    for (let lod = 0; lod < 2; lod++) {
      const cg = this.capGeos[lod];
      const vb = gl.createBuffer()!;
      gl.bindBuffer(gl.ARRAY_BUFFER, vb);
      gl.bufferData(gl.ARRAY_BUFFER, cg.interleaved, gl.STATIC_DRAW);
      this.capVBOs.push(vb);

      const ib = gl.createBuffer()!;
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, cg.indices, gl.STATIC_DRAW);
      this.capIBOs.push(ib);
      this.capIC.push(cg.indices.length);
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);

    // ── Instance buffers ──
    for (let lod = 0; lod < 3; lod++) {
      const buf = gl.createBuffer()!;
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0]), gl.DYNAMIC_DRAW);
      this.lodInstanceBufs.push(buf);
    }

    // Cap instance buffer
    this.capInstanceBuf = gl.createBuffer()!;

    // ── Build shaders ──
    this.bodyProgram = buildProgram(gl, bodyVertSrc, pbrFragSrc);
    this.capProgram = buildProgram(gl, capVertSrc, pbrFragSrc);
    this.shadowProgram = buildProgram(gl, shadowVertSrc, shadowFragSrc);

    // ── Set uniform block bindings ──
    for (const prog of [this.bodyProgram, this.capProgram]) {
      const camIdx = gl.getUniformBlockIndex(prog, 'Camera');
      const matIdx = gl.getUniformBlockIndex(prog, 'Material');
      console.log(`[WebGL] Program uniform blocks: Camera=${camIdx} Material=${matIdx} (INVALID=${gl.INVALID_INDEX})`);
      if (camIdx !== gl.INVALID_INDEX) gl.uniformBlockBinding(prog, camIdx, 0);
      if (matIdx !== gl.INVALID_INDEX) gl.uniformBlockBinding(prog, matIdx, 1);
    }

    // ── Get uniform locations ──
    const bp = this.bodyProgram;
    this.u_lightDir = gl.getUniformLocation(bp, 'lightDir')!;
    this.u_segTex = gl.getUniformLocation(bp, 'segTex')!;
    this.u_colorTex = gl.getUniformLocation(bp, 'colorTex')!;
    this.u_texWidth = gl.getUniformLocation(bp, 'texWidth')!;
    this.u_shadowTex = gl.getUniformLocation(bp, 'shadowTex')!;
    this.u_shadowVP = gl.getUniformLocation(bp, 'shadowVP')!;
    this.u_shadowSoftness = gl.getUniformLocation(bp, 'shadowSoftness')!;
    this.u_envTex = gl.getUniformLocation(bp, 'envTex')!;
    console.log(`[WebGL] Body program uniforms — lightDir:${!!this.u_lightDir} shadowTex:${!!this.u_shadowTex} shadowSoftness:${!!this.u_shadowSoftness} envTex:${!!this.u_envTex}`);

    const cp = this.capProgram;
    this.u_cap_lightDir = gl.getUniformLocation(cp, 'lightDir')!;
    this.u_cap_segTex = gl.getUniformLocation(cp, 'segTex')!;
    this.u_cap_colorTex = gl.getUniformLocation(cp, 'colorTex')!;
    this.u_cap_texWidth = gl.getUniformLocation(cp, 'texWidth')!;
    this.u_cap_shadowTex = gl.getUniformLocation(cp, 'shadowTex')!;
    this.u_cap_shadowVP = gl.getUniformLocation(cp, 'shadowVP')!;
    this.u_cap_shadowSoftness = gl.getUniformLocation(cp, 'shadowSoftness')!;
    this.u_cap_envTex = gl.getUniformLocation(cp, 'envTex')!;

    const sp = this.shadowProgram;
    this.u_shadow_shadowVP = gl.getUniformLocation(sp, 'shadowVP')!;
    this.u_shadow_segTex = gl.getUniformLocation(sp, 'segTex')!;
    this.u_shadow_texWidth = gl.getUniformLocation(sp, 'texWidth')!;

    // ── Uniform buffers (allocate once, update with bufferSubData) ──
    this.cameraUBO = gl.createBuffer()!;
    gl.bindBuffer(gl.UNIFORM_BUFFER, this.cameraUBO);
    gl.bufferData(gl.UNIFORM_BUFFER, new Float32Array(36), gl.DYNAMIC_DRAW); // 144 bytes

    this.materialUBO = gl.createBuffer()!;
    gl.bindBuffer(gl.UNIFORM_BUFFER, this.materialUBO);
    gl.bufferData(gl.UNIFORM_BUFFER, new Float32Array(12), gl.DYNAMIC_DRAW); // 48 bytes

    this.lightDirUBO = gl.createBuffer()!;
    gl.bindBuffer(gl.UNIFORM_BUFFER, this.lightDirUBO);
    gl.bufferData(gl.UNIFORM_BUFFER, new Float32Array(4), gl.DYNAMIC_DRAW); // 16 bytes
    gl.bindBuffer(gl.UNIFORM_BUFFER, null);

    // ── Shadow map ──
    this.shadowTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.shadowTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.DEPTH_COMPONENT24, this.shadowSize, this.shadowSize, 0,
      gl.DEPTH_COMPONENT, gl.UNSIGNED_INT, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_COMPARE_MODE, gl.COMPARE_REF_TO_TEXTURE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_COMPARE_FUNC, gl.LEQUAL);

    this.shadowFBO = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.shadowFBO);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, this.shadowTex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);

    // ── Write initial material UBO ──
    this.writeMaterialUBO();
    this.writeLightDirUBO();
  }

  setSegments(segTexs: SegmentTextures, data: SegbinData) {
    this.segmentTextures = segTexs;
    this.segbinData = data;

    // Upload cap instance buffer
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.capInstanceBuf);
    gl.bufferData(gl.ARRAY_BUFFER, segTexs.capInstances, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  setModelBounds(cx: number, cy: number, cz: number, rx: number, ry: number, rz: number) {
    this._modelCenter = [cx, cy, cz];
    this._modelRadius = Math.max(rx, ry, rz);
  }

  /** Compute shadow view-projection matrix from light direction. */
  computeShadowVP(lx: number, ly: number, lz: number): Float32Array {
    const [cx, cy, cz] = this._modelCenter;
    const r = this._modelRadius * 1.5;

    // Light position: center - lightDir * radius
    const lp = [cx - lx * r, cy - ly * r, cz - lz * r];

    // View matrix: lookAt(lp, center, up=Z)
    const fwd = [cx - lp[0], cy - lp[1], cz - lp[2]];
    const fwdLen = Math.sqrt(fwd[0] ** 2 + fwd[1] ** 2 + fwd[2] ** 2);
    fwd[0] /= fwdLen; fwd[1] /= fwdLen; fwd[2] /= fwdLen;

    const right = [fwd[1], -fwd[0], 0];
    const rightLen = Math.sqrt(right[0] ** 2 + right[1] ** 2);
    if (rightLen > 0.001) { right[0] /= rightLen; right[1] /= rightLen; }
    else { right[0] = 1; right[1] = 0; }

    const up = [
      right[1] * fwd[2] - right[2] * fwd[1],
      right[2] * fwd[0] - right[0] * fwd[2],
      right[0] * fwd[1] - right[1] * fwd[0],
    ];

    // Column-major view matrix
    const view = new Float32Array(16);
    view[0] = right[0]; view[1] = up[0]; view[2] = -fwd[0]; view[3] = 0;
    view[4] = right[1]; view[5] = up[1]; view[6] = -fwd[1]; view[7] = 0;
    view[8] = right[2]; view[9] = up[2]; view[10] = -fwd[2]; view[11] = 0;
    view[12] = -(right[0] * lp[0] + right[1] * lp[1] + right[2] * lp[2]);
    view[13] = -(up[0] * lp[0] + up[1] * lp[1] + up[2] * lp[2]);
    view[14] = fwd[0] * lp[0] + fwd[1] * lp[1] + fwd[2] * lp[2];
    view[15] = 1;

    // Orthographic projection (column-major)
    const proj = new Float32Array(16);
    const s = r;
    proj[0] = 1 / s; proj[1] = 0; proj[2] = 0; proj[3] = 0;
    proj[4] = 0; proj[5] = 1 / s; proj[6] = 0; proj[7] = 0;
    proj[8] = 0; proj[9] = 0; proj[10] = -2 / (r * 4); proj[11] = 0;
    proj[12] = 0; proj[13] = 0; proj[14] = 0; proj[15] = 1;

    // Multiply: shadowVP = proj × view
    const vp = new Float32Array(16);
    for (let col = 0; col < 4; col++) {
      for (let row = 0; row < 4; row++) {
        vp[col * 4 + row] =
          proj[row] * view[col * 4] +
          proj[4 + row] * view[col * 4 + 1] +
          proj[8 + row] * view[col * 4 + 2] +
          proj[12 + row] * view[col * 4 + 3];
      }
    }
    return vp;
  }

  writeCameraUBO(camera: OrbitCamera) {
    const gl = this.gl;
    // Camera UBO layout (std140):
    //   mat4 viewProj (64 bytes, offset 0)
    //   mat4 viewMat  (64 bytes, offset 64)
    //   vec3 camPos   (12 bytes, offset 128, padded to 16)
    // Total: 144 bytes = 36 floats
    const data = new Float32Array(36);
    data.set(camera.viewProj, 0);   // offset 0
    data.set(camera.viewMat, 16);   // offset 64 → 16 floats
    data[32] = camera.position[0];  // offset 128
    data[33] = camera.position[1];
    data[34] = camera.position[2];
    // data[35] = padding

    gl.bindBuffer(gl.UNIFORM_BUFFER, this.cameraUBO);
    gl.bufferSubData(gl.UNIFORM_BUFFER, 0, data);
    gl.bindBuffer(gl.UNIFORM_BUFFER, null);
  }

  private _materialWriteCount = 0;

  writeMaterialUBO() {
    const gl = this.gl;
    this._materialWriteCount++;
    if (this._materialWriteCount <= 3 || this._materialWriteCount % 60 === 0) {
      console.log(`[WebGL] writeMaterialUBO #${this._materialWriteCount} roughness=${this.material.roughness} metalness=${this.material.metalness} useRole=${this.material.useRoleColors} tint=[${this.material.baseColorTint.join(',')}]`);
    }
    // Material UBO layout (std140):
    //   float roughness      @ 0
    //   float metalness      @ 4
    //   float envIntensity   @ 8
    //   float specularStrength @ 12
    //   float ambientStrength @ 16
    //   float useRoleColors   @ 20
    //   (8 bytes padding)
    //   vec3 baseColorTint    @ 32  (16-byte aligned)
    // Total: 44 → padded to 48 bytes = 12 floats
    const data = new Float32Array(12);
    data[0] = this.material.roughness;
    data[1] = this.material.metalness;
    data[2] = this.material.envIntensity;
    data[3] = this.material.specularStrength;
    data[4] = this.material.ambientStrength;
    data[5] = this.material.useRoleColors;
    // data[6], data[7] = padding
    data[8] = this.material.baseColorTint[0];
    data[9] = this.material.baseColorTint[1];
    data[10] = this.material.baseColorTint[2];
    // data[11] = padding (48-byte total)

    gl.bindBuffer(gl.UNIFORM_BUFFER, this.materialUBO);
    gl.bufferSubData(gl.UNIFORM_BUFFER, 0, data);
    gl.bindBuffer(gl.UNIFORM_BUFFER, null);
  }

  writeLightDirUBO() {
    const gl = this.gl;
    const data = new Float32Array(4);
    data[0] = this.lightDir[0];
    data[1] = this.lightDir[1];
    data[2] = this.lightDir[2];
    data[3] = this.lightDir[3];

    gl.bindBuffer(gl.UNIFORM_BUFFER, this.lightDirUBO);
    gl.bufferSubData(gl.UNIFORM_BUFFER, 0, data);
    gl.bindBuffer(gl.UNIFORM_BUFFER, null);
  }

  // ── Load environment map ──
  private _envLinearFilter: boolean | null = null;

  async setEnvMap(url: string): Promise<void> {
    const gl = this.gl;

    // Check float linear filtering support once
    if (this._envLinearFilter === null) {
      this._envLinearFilter = !!gl.getExtension('OES_texture_float_linear');
    }

    const hdr = await loadHDR(url);

    // Convert Float16Array → Float32Array for WebGL upload
    const f32 = new Float32Array(hdr.data.length);
    for (let i = 0; i < hdr.data.length; i++) {
      f32[i] = hdr.data[i];
    }

    const filter = this._envLinearFilter ? gl.LINEAR : gl.NEAREST;

    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, hdr.width, hdr.height, 0, gl.RGBA, gl.FLOAT, f32);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);

    if (this.envTex) gl.deleteTexture(this.envTex);
    this.envTex = tex;
  }

  // ── Render methods ──

  /** Evaluate LOD and upload instance buffers. Call once per frame before drawing. */
  updateLOD(camera: OrbitCamera, canvas: HTMLCanvasElement) {
    const [lod0, lod1, lod2] = evaluateLOD(this.segbinData, camera, canvas.width, canvas.height);
    const gl = this.gl;

    const upload = (buf: WebGLBuffer, arr: Float32Array) => {
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER, arr, gl.DYNAMIC_DRAW);
    };
    upload(this.lodInstanceBufs[0], lod0);
    upload(this.lodInstanceBufs[1], lod1);
    upload(this.lodInstanceBufs[2], lod2);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    // Store for stats readback
    this._lodCounts = [lod0.length, lod1.length, lod2.length];

    return { lod0: lod0.length, lod1: lod1.length, lod2: lod2.length };
  }

  /** Per-frame LOD instance counts (set by updateLOD, read by stats). */
  _lodCounts: [number, number, number] = [0, 0, 0];

  /** Render shadow map from light's POV. */
  renderShadowMap(camera: OrbitCamera, canvasWidth: number, canvasHeight: number) {
    const gl = this.gl;
    const { segTex, texWidth } = this.segmentTextures;

    // LOD eval for shadow pass (only LOD 0 + 1 cast shadows)
    const [lod0, lod1, _lod2] = evaluateLOD(this.segbinData, camera, canvasWidth, canvasHeight);

    const shadowVP = this.computeShadowVP(this.lightDir[0], this.lightDir[1], this.lightDir[2]);

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.shadowFBO);
    gl.viewport(0, 0, this.shadowSize, this.shadowSize);
    gl.clear(gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);

    gl.useProgram(this.shadowProgram);
    gl.uniformMatrix4fv(this.u_shadow_shadowVP, false, shadowVP);
    gl.uniform1i(this.u_shadow_segTex, 0);
    gl.uniform1i(this.u_shadow_texWidth, texWidth);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, segTex);

    // Draw both LOD 0 and 1
    for (let lod = 0; lod < 2; lod++) {
      const indices = lod === 0 ? lod0 : lod1;
      if (indices.length === 0) continue;

      // Upload instance buffer
      gl.bindBuffer(gl.ARRAY_BUFFER, this.lodInstanceBufs[lod]);
      gl.bufferData(gl.ARRAY_BUFFER, indices, gl.DYNAMIC_DRAW);

      this._bindShadowVAO(lod, this.lodInstanceBufs[lod]);
      gl.drawElementsInstanced(gl.TRIANGLES, this.bodyIC[lod], gl.UNSIGNED_SHORT, 0, indices.length);
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /** Render body segments (all 3 LOD levels) to the current FBO. */
  drawBody(camera: OrbitCamera, canvas: HTMLCanvasElement) {
    const gl = this.gl;
    const { segTex, colorTex, texWidth } = this.segmentTextures;

    const counts = this.updateLOD(camera, canvas);

    gl.useProgram(this.bodyProgram);

    // Bind UBOs
    this._bindUBO(0, this.cameraUBO);
    this._bindUBO(1, this.materialUBO);

    // Set uniforms
    gl.uniform4f(this.u_lightDir, this.lightDir[0], this.lightDir[1], this.lightDir[2], this.lightDir[3]);
    gl.uniform1i(this.u_segTex, 0);
    gl.uniform1i(this.u_colorTex, 1);
    gl.uniform1i(this.u_texWidth, texWidth);
    gl.uniform1i(this.u_shadowTex, 2);
    gl.uniformMatrix4fv(this.u_shadowVP, false, this.computeShadowVP(this.lightDir[0], this.lightDir[1], this.lightDir[2]));
    gl.uniform1f(this.u_shadowSoftness, this.shadowSoftness);
    gl.uniform1i(this.u_envTex, 3);

    // Bind textures
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, segTex);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, colorTex);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.shadowTex);
    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, this.envTex);

    for (let lod = 0; lod < 3; lod++) {
      const count = lod === 0 ? counts.lod0 : lod === 1 ? counts.lod1 : counts.lod2;
      if (count === 0) continue;

      this._bindBodyVAO(lod, this.lodInstanceBufs[lod]);
      gl.drawElementsInstanced(gl.TRIANGLES, this.bodyIC[lod], gl.UNSIGNED_SHORT, 0, count);
    }

    // Unbind textures
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  /** Render endcaps to the current FBO. */
  drawCaps() {
    const gl = this.gl;
    const { segTex, colorTex, texWidth, capCount } = this.segmentTextures;
    if (capCount === 0) return;

    gl.useProgram(this.capProgram);

    // Bind UBOs
    this._bindUBO(0, this.cameraUBO);
    this._bindUBO(1, this.materialUBO);

    // Set uniforms
    gl.uniform4f(this.u_cap_lightDir, this.lightDir[0], this.lightDir[1], this.lightDir[2], this.lightDir[3]);
    gl.uniform1i(this.u_cap_segTex, 0);
    gl.uniform1i(this.u_cap_colorTex, 1);
    gl.uniform1i(this.u_cap_texWidth, texWidth);
    gl.uniform1i(this.u_cap_shadowTex, 2);
    gl.uniformMatrix4fv(this.u_cap_shadowVP, false, this.computeShadowVP(this.lightDir[0], this.lightDir[1], this.lightDir[2]));
    gl.uniform1f(this.u_cap_shadowSoftness, this.shadowSoftness);
    gl.uniform1i(this.u_cap_envTex, 3);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, segTex);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, colorTex);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.shadowTex);
    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, this.envTex);

    for (let lod = 0; lod < 2; lod++) {
      this._bindCapVAO(lod);
      gl.drawElementsInstanced(gl.TRIANGLES, this.capIC[lod], gl.UNSIGNED_SHORT, 0, capCount);
    }

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  // ── VAO helpers ──
  private _bodyVAOs: (WebGLVertexArrayObject | null)[] = [null, null, null];

  private _bindBodyVAO(lod: number, instanceBuf: WebGLBuffer) {
    const gl = this.gl;
    if (!this._bodyVAOs[lod]) {
      const vao = gl.createVertexArray()!;
      gl.bindVertexArray(vao);
      // Vertex attributes: position (0), normal (1)
      gl.bindBuffer(gl.ARRAY_BUFFER, this.bodyVBOs[lod]);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 24, 0); // 3 pos floats
      gl.enableVertexAttribArray(1);
      gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 24, 12); // 3 normal floats
      // Instanced attribute: segIndex (2)
      gl.bindBuffer(gl.ARRAY_BUFFER, instanceBuf);
      gl.enableVertexAttribArray(2);
      gl.vertexAttribPointer(2, 1, gl.FLOAT, false, 0, 0);
      gl.vertexAttribDivisor(2, 1);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.bodyIBOs[lod]);
      gl.bindVertexArray(null);
      this._bodyVAOs[lod] = vao;
    }
    gl.bindVertexArray(this._bodyVAOs[lod]);
    // Re-bind instance buffer (may have been updated)
    gl.bindBuffer(gl.ARRAY_BUFFER, instanceBuf);
    gl.vertexAttribPointer(2, 1, gl.FLOAT, false, 0, 0);
  }

  private _capVAOs: (WebGLVertexArrayObject | null)[] = [null, null];

  private _bindCapVAO(lod: number) {
    const gl = this.gl;
    if (!this._capVAOs[lod]) {
      const vao = gl.createVertexArray()!;
      gl.bindVertexArray(vao);
      // Vertex attributes: position (0), normal (1)
      gl.bindBuffer(gl.ARRAY_BUFFER, this.capVBOs[lod]);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 24, 0);
      gl.enableVertexAttribArray(1);
      gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 24, 12);
      // Instanced: segIndex (2), isEnd (3)
      gl.bindBuffer(gl.ARRAY_BUFFER, this.capInstanceBuf);
      gl.enableVertexAttribArray(2);
      gl.vertexAttribPointer(2, 1, gl.FLOAT, false, 8, 0);
      gl.vertexAttribDivisor(2, 1);
      gl.enableVertexAttribArray(3);
      gl.vertexAttribPointer(3, 1, gl.FLOAT, false, 8, 4);
      gl.vertexAttribDivisor(3, 1);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.capIBOs[lod]);
      gl.bindVertexArray(null);
      this._capVAOs[lod] = vao;
    }
    gl.bindVertexArray(this._capVAOs[lod]);
  }

  private _shadowVAOs: (WebGLVertexArrayObject | null)[] = [null, null, null];

  private _bindShadowVAO(lod: number, instanceBuf: WebGLBuffer) {
    const gl = this.gl;
    if (!this._shadowVAOs[lod]) {
      const vao = gl.createVertexArray()!;
      gl.bindVertexArray(vao);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.bodyVBOs[lod]);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 24, 0);
      gl.enableVertexAttribArray(1);
      gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 24, 12);
      gl.bindBuffer(gl.ARRAY_BUFFER, instanceBuf);
      gl.enableVertexAttribArray(2);
      gl.vertexAttribPointer(2, 1, gl.FLOAT, false, 0, 0);
      gl.vertexAttribDivisor(2, 1);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.bodyIBOs[lod]);
      gl.bindVertexArray(null);
      this._shadowVAOs[lod] = vao;
    }
    gl.bindVertexArray(this._shadowVAOs[lod]);
    gl.bindBuffer(gl.ARRAY_BUFFER, instanceBuf);
    gl.vertexAttribPointer(2, 1, gl.FLOAT, false, 0, 0);
  }

  private _bindUBO(index: number, buf: WebGLBuffer) {
    const gl = this.gl;
    gl.bindBufferBase(gl.UNIFORM_BUFFER, index, buf);
  }

  dispose() {
    const gl = this.gl;
    gl.deleteProgram(this.bodyProgram);
    gl.deleteProgram(this.capProgram);
    gl.deleteProgram(this.shadowProgram);
    for (const b of this.bodyVBOs) gl.deleteBuffer(b);
    for (const b of this.bodyIBOs) gl.deleteBuffer(b);
    for (const b of this.capVBOs) gl.deleteBuffer(b);
    for (const b of this.capIBOs) gl.deleteBuffer(b);
    for (const b of this.lodInstanceBufs) gl.deleteBuffer(b);
    gl.deleteBuffer(this.capInstanceBuf);
    for (const v of this._bodyVAOs) if (v) gl.deleteVertexArray(v);
    for (const v of this._capVAOs) if (v) gl.deleteVertexArray(v);
    for (const v of this._shadowVAOs) if (v) gl.deleteVertexArray(v);
    gl.deleteBuffer(this.cameraUBO);
    gl.deleteBuffer(this.materialUBO);
    gl.deleteBuffer(this.lightDirUBO);
    gl.deleteFramebuffer(this.shadowFBO);
    gl.deleteTexture(this.shadowTex);
    if (this.envTex) gl.deleteTexture(this.envTex);
    if (this.segmentTextures) {
      gl.deleteTexture(this.segmentTextures.segTex);
      gl.deleteTexture(this.segmentTextures.colorTex);
    }
  }
}

// ── Shader compilation helpers ──

function compileShader(gl: WebGL2RenderingContext, src: string, type: number): WebGLShader {
  const shader = gl.createShader(type)!;
  gl.shaderSource(shader, src);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`Shader compile error:\n${log}\n\nSource:\n${src.slice(0, 2000)}`);
  }
  return shader;
}

function buildProgram(gl: WebGL2RenderingContext, vertSrc: string, fragSrc: string): WebGLProgram {
  const vs = compileShader(gl, vertSrc, gl.VERTEX_SHADER);
  const fs = compileShader(gl, fragSrc, gl.FRAGMENT_SHADER);
  const prog = gl.createProgram()!;
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(prog);
    throw new Error(`Program link error:\n${log}`);
  }
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  return prog;
}
