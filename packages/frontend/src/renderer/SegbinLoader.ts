import { Mesh, VertexData, ShaderMaterial, RawTexture, Texture, Matrix, Vector2, Vector3, Engine, Scene } from '@babylonjs/core';
import type { SegbinData } from './types';
import { roleColor } from './types';
export { Role } from './types';
import { LOD_BODY_GEO, LOD_CAP_GEO } from './geometry';
import { SEGMENT_VERTEX_SHADER, ENDCAP_VERTEX_SHADER, SEGMENT_FRAGMENT_SHADER } from './shaders';

export type { SegbinData } from './types';

// ── Segment direction helper ────────────────────────────────────────────

function segDir(g: Float32Array, i: number): [number, number, number] {
  const dx = g[i * 8 + 3] - g[i * 8], dy = g[i * 8 + 4] - g[i * 8 + 1], dz = g[i * 8 + 5] - g[i * 8 + 2];
  const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
  if (len < 0.001) return [0, 0, 1];
  return [dx / len, dy / len, dz / len];
}

// ── Shared segment-data texture ─────────────────────────────────────────

function makeSharedTexture(
  data: SegbinData,
  scene: Scene,
  excludeRoles?: Set<number>,
): { tex: RawTexture; texWidth: number; texHeight: number } {
  const TEX_WIDTH = 2048;
  const totalTexels = data.count * 4;
  const texHeight = Math.ceil(totalTexels / TEX_WIDTH);
  const texels = new Float32Array(TEX_WIDTH * texHeight * 4);
  const g = data.geoms;
  const cc = data.chainContinue;
  const st = data.segType;

  for (let i = 0; i < data.count; i++) {
    const sx = g[i * 8], sy = g[i * 8 + 1], sz = g[i * 8 + 2];
    const ex = g[i * 8 + 3], ey = g[i * 8 + 4], ez = g[i * 8 + 5];
    let width = g[i * 8 + 6];
    const layerZ = g[i * 8 + 7];
    const base = i * 16;
    const isArc = st[i] === 1;

    if (excludeRoles?.has(data.roles[i])) width = 0;

    if (isArc) {
      const arcWidth = g[i * 8 + 6];
      const packed = g[i * 8 + 7];
      const layerZArc = Math.round(packed * 100) / 100;
      const conicWeight = (packed - layerZArc) * 10000;
      texels[base + 0] = sx; texels[base + 1] = sy;
      texels[base + 2] = sz; texels[base + 3] = arcWidth;
      texels[base + 4] = ex; texels[base + 5] = ey;
      texels[base + 6] = ez; texels[base + 7] = conicWeight;
      texels[base + 11] = layerZArc;
    } else {
      texels[base + 0] = sx; texels[base + 1] = sy;
      texels[base + 2] = sz; texels[base + 3] = layerZ;
      texels[base + 4] = ex; texels[base + 5] = ey;
      texels[base + 6] = ez; texels[base + 7] = width;
    }

    const startCap = (i === 0 || cc[i - 1] === 0) ? 1 : 0;
    const endCap   = (i === data.count - 1 || cc[i] === 0) ? 1 : 0;
    texels[base + 8]  = isArc ? 0 : startCap;
    texels[base + 9]  = isArc ? 0 : endCap;
    texels[base + 10] = isArc ? 1 : 0; // row2.z = seg_type flag for shader
    texels[base + 11] = texels[base + 11] || 0; // preserve arc layerZ

    // Row 3: chain-start tangent
    let ctx: number, cty: number, ctz: number;
    if (i > 0 && cc[i - 1] === 1) {
      if (st[i - 1] === 1) {
        const p1x = g[(i-1)*8 + 3], p1y = g[(i-1)*8 + 4], p1z = g[(i-1)*8 + 5];
        const p2x = g[i*8], p2y = g[i*8 + 1], p2z = g[i*8 + 2];
        const dx = p2x - p1x, dy = p2y - p1y, dz = p2z - p1z;
        const len = Math.sqrt(dx*dx + dy*dy + dz*dz);
        if (len > 0.001) {
          ctx = dx / len; cty = dy / len; ctz = dz / len;
        } else {
          [ctx, cty, ctz] = segDir(g, i);
        }
      } else {
        [ctx, cty, ctz] = segDir(g, i - 1);
      }
    } else {
      [ctx, cty, ctz] = segDir(g, i);
    }
    texels[base + 12] = ctx;
    texels[base + 13] = cty;
    texels[base + 14] = ctz;
    texels[base + 15] = 0;
  }

  const tex = new RawTexture(
    texels,
    TEX_WIDTH,
    texHeight,
    Engine.TEXTUREFORMAT_RGBA,
    scene,
    false,  // no mipmaps
    false,  // no invertY
    Texture.NEAREST_SAMPLINGMODE,
    Engine.TEXTURETYPE_FLOAT,
  );
  tex.wrapU = Texture.CLAMP_ADDRESSMODE;
  tex.wrapV = Texture.CLAMP_ADDRESSMODE;
  return { tex, texWidth: TEX_WIDTH, texHeight };
}

// ── LOD mesh group ───────────────────────────────────────────────────────

export interface LODMeshGroup {
  body: Mesh;
  caps: Mesh | null;  // null for LOD 2 (no caps)
  lod: number;        // 0, 1, or 2
  /** Per-instance visibility flag (1.0 = visible, 0.0 = culled) */
  bodyVis: Float32Array;
  /** Per-instance visibility for caps (null for LOD 2) */
  capsVis: Float32Array | null;
  /** For each cap instance, which segment index it belongs to */
  capSegIdx: Uint32Array | null;
}

function buildOneLOD(
  data: SegbinData,
  scene: Scene,
  tex: RawTexture,
  texWidth: number,
  texHeight: number,
  lod: number,
  activeCaps: number,
  capTex: RawTexture | null,
  capTexelW: number,
  capTexelH: number,
  dummyShadow: RawTexture,
  dummyEnv: RawTexture,
): LODMeshGroup {
  const count = data.count;
  const identity = Matrix.Identity();
  const bodyGeo = LOD_BODY_GEO[lod];
  const isLod2 = lod === 2;

  // ── Body mesh ──────────────────────────────────────────────────────────
  const bodyMesh = new Mesh(`body_LOD${lod}`, scene);
  const bodyVD = new VertexData();
  bodyVD.positions = bodyGeo.positions;
  bodyVD.normals = bodyGeo.normals;
  bodyVD.indices = bodyGeo.indices;
  bodyVD.applyToMesh(bodyMesh);

  const bodyMat = new ShaderMaterial(`bodyMat_LOD${lod}`, scene, {
    vertexSource: SEGMENT_VERTEX_SHADER,
    fragmentSource: SEGMENT_FRAGMENT_SHADER,
  }, {
    attributes: ['position', 'normal', 'instColor', 'instVisible'],
    uniforms: [
      'worldViewProjection', 'view',
      'uSegmentCount', 'uHeightScale', 'uAreaCorrection',
      'uTexSize', 'uRoughness', 'uMetalness', 'uKeyLightDir', 'uCameraPos',
      'uShadowMatrix', 'uShadowMapSize', 'uEnvMapLOD', 'uEnvIntensity',
      'uBaseColorTint', 'uSpecularStrength', 'uAmbientStrength', 'uLodMode', 'uDbgNormFlip', 'uDbgHighlight', 'uDbgNormVis',
    ],
    samplers: ['segmentTexture', 'uShadowMap', 'uEnvMapEQ'],
    needAlphaBlending: false,
  });
  bodyMat.backFaceCulling = false;
  bodyMat.setTexture('segmentTexture', tex);
  bodyMat.setTexture('uShadowMap', dummyShadow);
  bodyMat.setTexture('uEnvMapEQ', dummyEnv);
  bodyMat.setFloat('uSegmentCount', count);
  bodyMat.setFloat('uHeightScale', 1.25);
  bodyMat.setFloat('uAreaCorrection', 1.1);
  bodyMat.setFloat('uRoughness', 0.10);
  bodyMat.setFloat('uMetalness', 0.0);
  bodyMat.setVector3('uKeyLightDir', new Vector3(0.416, -0.25, 0.872));
  bodyMat.setVector2('uTexSize', new Vector2(texWidth, texHeight));
  bodyMat.setVector2('uShadowMapSize', new Vector2(4096, 4096));
  bodyMat.setFloat('uEnvMapLOD', 0.0);
  bodyMat.setFloat('uEnvIntensity', 0.25);
  bodyMat.setVector3('uBaseColorTint', new Vector3(1, 1, 1));
  bodyMat.setFloat('uSpecularStrength', 1.0);
  bodyMat.setFloat('uAmbientStrength', 0.5);
  bodyMat.setFloat('uLodMode', isLod2 ? 1.0 : 0.0);
  bodyMat.setFloat('uDbgNormFlip', 0.0);
  bodyMat.setFloat('uDbgHighlight', 0.0);
  bodyMat.setFloat('uDbgNormVis', 0.0);

  bodyMesh.material = bodyMat;
  bodyMesh.alwaysSelectAsActiveMesh = true;

  bodyMesh.thinInstanceRegisterAttribute('instColor', 4);
  for (let j = 0; j < count; j++) {
    bodyMesh.thinInstanceAdd(identity, false);
  }
  const bodyColors = new Float32Array(count * 4);
  for (let j = 0; j < count; j++) {
    const c = roleColor(data.roles[j]);
    bodyColors[j * 4] = c.r;
    bodyColors[j * 4 + 1] = c.g;
    bodyColors[j * 4 + 2] = c.b;
    bodyColors[j * 4 + 3] = 1.0;
  }
  bodyMesh.thinInstanceSetBuffer('instColor', bodyColors, 4);

  // Per-instance visibility (updated each frame by render loop)
  // Start with only LOD 0 visible; others hidden until first frame's LOD update
  const bodyVis = new Float32Array(count * 4);
  if (lod === 0) bodyVis.fill(1.0, 0, count * 4);
  bodyMesh.thinInstanceRegisterAttribute('instVisible', 4);
  bodyMesh.thinInstanceSetBuffer('instVisible', bodyVis, 4);

  // ── Cap mesh (LOD 2 has no caps) ───────────────────────────────────────
  if (isLod2) {
    return { body: bodyMesh, caps: null, lod, bodyVis, capsVis: null, capSegIdx: null };
  }

  const capGeo = LOD_CAP_GEO[lod];
  const capMesh = new Mesh(`caps_LOD${lod}`, scene);
  const capVD = new VertexData();
  capVD.positions = capGeo.positions;
  capVD.normals = capGeo.normals;
  capVD.indices = capGeo.indices;
  capVD.applyToMesh(capMesh);

  const capMat = new ShaderMaterial(`capMat_LOD${lod}`, scene, {
    vertexSource: ENDCAP_VERTEX_SHADER,
    fragmentSource: SEGMENT_FRAGMENT_SHADER,
  }, {
    attributes: ['position', 'normal', 'instColor', 'instVisible'],
    uniforms: [
      'worldViewProjection', 'view',
      'uSegmentCount', 'uHeightScale', 'uAreaCorrection',
      'uTexSize', 'uCapTexSize', 'uRoughness', 'uMetalness',
      'uKeyLightDir', 'uCameraPos',
      'uShadowMatrix', 'uShadowMapSize', 'uEnvMapLOD', 'uEnvIntensity',
      'uBaseColorTint', 'uSpecularStrength', 'uAmbientStrength', 'uLodMode', 'uDbgHighlight', 'uDbgNormVis',
    ],
    samplers: ['segmentTexture', 'capMapTexture', 'uShadowMap', 'uEnvMapEQ'],
    needAlphaBlending: false,
  });
  capMat.backFaceCulling = false;
  capMat.setTexture('segmentTexture', tex);
  capMat.setTexture('capMapTexture', capTex!);
  capMat.setTexture('uShadowMap', dummyShadow);
  capMat.setTexture('uEnvMapEQ', dummyEnv);
  capMat.setFloat('uSegmentCount', count);
  capMat.setFloat('uHeightScale', 1.25);
  capMat.setFloat('uAreaCorrection', 1.1);
  capMat.setFloat('uRoughness', 0.10);
  capMat.setFloat('uMetalness', 0.0);
  capMat.setVector3('uKeyLightDir', new Vector3(0.416, -0.25, 0.872));
  capMat.setVector2('uTexSize', new Vector2(texWidth, texHeight));
  capMat.setVector2('uCapTexSize', new Vector2(capTexelW, capTexelH));
  capMat.setVector2('uShadowMapSize', new Vector2(4096, 4096));
  capMat.setFloat('uEnvMapLOD', 0.0);
  capMat.setFloat('uEnvIntensity', 0.25);
  capMat.setVector3('uBaseColorTint', new Vector3(1, 1, 1));
  capMat.setFloat('uSpecularStrength', 1.0);
  capMat.setFloat('uAmbientStrength', 0.5);
  capMat.setFloat('uLodMode', 0.0);
  capMat.setFloat('uDbgHighlight', 0.0);
  capMat.setFloat('uDbgNormVis', 0.0);

  capMesh.material = capMat;
  capMesh.alwaysSelectAsActiveMesh = true;

  capMesh.thinInstanceRegisterAttribute('instColor', 4);
  for (let j = 0; j < activeCaps; j++) {
    capMesh.thinInstanceAdd(identity, false);
  }
  console.log(`[caps LOD${lod}] ${activeCaps} instances, ${capGeo.indices.length / 3} tris`);
  const capColors = new Float32Array(activeCaps * 4);
  const capSegIdx = new Uint32Array(activeCaps);
  const cc = data.chainContinue;
  let capN = 0;
  for (let i = 0; i < count; i++) {
    if (i === 0 || cc[i - 1] === 0) {
      const c = roleColor(data.roles[i]);
      capColors[capN * 4] = c.r; capColors[capN * 4 + 1] = c.g;
      capColors[capN * 4 + 2] = c.b; capColors[capN * 4 + 3] = 1.0;
      capSegIdx[capN] = i;
      capN++;
    }
    if (i === count - 1 || cc[i] === 0) {
      const c = roleColor(data.roles[i]);
      capColors[capN * 4] = c.r; capColors[capN * 4 + 1] = c.g;
      capColors[capN * 4 + 2] = c.b; capColors[capN * 4 + 3] = 1.0;
      capSegIdx[capN] = i;
      capN++;
    }
  }
  capMesh.thinInstanceSetBuffer('instColor', capColors, 4);

  const capsVis = new Float32Array(activeCaps * 4);
  if (lod === 0) capsVis.fill(1.0, 0, activeCaps * 4);
  capMesh.thinInstanceRegisterAttribute('instVisible', 4);
  capMesh.thinInstanceSetBuffer('instVisible', capsVis, 4);

  return { body: bodyMesh, caps: capMesh, lod, bodyVis, capsVis, capSegIdx };
}

// ── Build all LOD meshes ─────────────────────────────────────────────────

export interface BuildResult {
  /** All meshes in a flat list (for uniform iteration) */
  meshes: Mesh[];
  /** Per-LOD groups */
  groups: LODMeshGroup[];
  /** Shared textures */
  tex: RawTexture;
  capTex: RawTexture | null;
  /** Texture dimensions for the segment data texture */
  texWidth: number;
  texHeight: number;
  /** Total triangles for each LOD level (body + active caps) */
  trisPerLod: [number, number, number];
}

export function buildShaderMeshes(
  data: SegbinData,
  scene: Scene,
  excludeRoles?: Set<number>,
): BuildResult {
  const { tex, texWidth, texHeight } = makeSharedTexture(data, scene, excludeRoles);
  const count = data.count;
  const cc = data.chainContinue;

  // Count active caps
  let activeCaps = 0;
  for (let i = 0; i < count; i++) {
    if (i === 0 || cc[i - 1] === 0) activeCaps++;
    if (i === count - 1 || cc[i] === 0) activeCaps++;
  }

  // Cap map texture for LOD 0 / LOD 1
  const capTexelW = 2048;
  const capTexelH = Math.max(1, Math.ceil(activeCaps / capTexelW));
  const capMap = new Float32Array(capTexelW * capTexelH * 4);
  let capN = 0;
  for (let i = 0; i < count; i++) {
    if (i === 0 || cc[i - 1] === 0) {
      capMap[capN * 4] = i; capMap[capN * 4 + 1] = 0; capN++;
    }
    if (i === count - 1 || cc[i] === 0) {
      capMap[capN * 4] = i; capMap[capN * 4 + 1] = 1; capN++;
    }
  }
  for (let j = capN * 4; j < capMap.length; j++) capMap[j] = 0;

  const capTex = new RawTexture(capMap, capTexelW, capTexelH,
    Engine.TEXTUREFORMAT_RGBA, scene, false, false,
    Texture.NEAREST_SAMPLINGMODE, Engine.TEXTURETYPE_FLOAT);
  capTex.wrapU = Texture.CLAMP_ADDRESSMODE;
  capTex.wrapV = Texture.CLAMP_ADDRESSMODE;

  // Dummy fallback textures
  const dummyShadow = new RawTexture(new Float32Array(4), 1, 1,
    Engine.TEXTUREFORMAT_RGBA, scene, false, false,
    Texture.NEAREST_SAMPLINGMODE, Engine.TEXTURETYPE_FLOAT);
  const dummyEnv = new RawTexture(new Float32Array(4), 1, 1,
    Engine.TEXTUREFORMAT_RGBA, scene, false, false,
    Texture.NEAREST_SAMPLINGMODE, Engine.TEXTURETYPE_FLOAT);

  // Print triangle stats
  for (let lod = 0; lod <= 2; lod++) {
    const bodyTris = LOD_BODY_GEO[lod].indices.length / 3;
    const capTris = LOD_CAP_GEO[lod].indices.length / 3;
    console.log(`[LOD ${lod}] body: ${(bodyTris * count).toLocaleString()} tris, caps: ${(capTris * activeCaps).toLocaleString()} tris`);
  }

  // Build each LOD level
  const groups: LODMeshGroup[] = [];
  const meshes: Mesh[] = [];
  for (let lod = 0; lod <= 2; lod++) {
    const g = buildOneLOD(data, scene, tex, texWidth, texHeight, lod,
      activeCaps, capTex, capTexelW, capTexelH, dummyShadow, dummyEnv);
    groups.push(g);
    meshes.push(g.body);
    if (g.caps) meshes.push(g.caps);
  }

  const trisPerLod: [number, number, number] = [
    (LOD_BODY_GEO[0].indices.length / 3) * count + (LOD_CAP_GEO[0].indices.length / 3) * activeCaps,
    (LOD_BODY_GEO[1].indices.length / 3) * count + (LOD_CAP_GEO[1].indices.length / 3) * activeCaps,
    (LOD_BODY_GEO[2].indices.length / 3) * count,
  ];

  return { meshes, groups, tex, capTex, texWidth, texHeight, trisPerLod };
}

export async function loadSegbin(url: string): Promise<SegbinData> {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`Failed to fetch ${url}: ${resp.status}`);
  const buf = await resp.arrayBuffer();
  return parseSegbin(buf);
}

export function parseSegbin(buf: ArrayBuffer): SegbinData {
  const dv = new DataView(buf);

  const magic = dv.getUint32(0, true);
  if (magic !== 0x31474553) {
    throw new Error(`Bad segbin magic: 0x${magic.toString(16)}`);
  }
  const version = dv.getUint16(4, true);
  if (version !== 1) {
    throw new Error(`Unsupported segbin version: ${version}`);
  }
  const count = dv.getUint32(8, true);
  const flags = dv.getUint32(12, true);

  const geoms = new Float32Array(buf, 16, count * 8);
  let offset = 16 + count * 32;
  const roles = new Uint8Array(buf, offset, count);
  offset += count;

  let chainContinue: Uint8Array;
  if (flags & 4) {
    chainContinue = new Uint8Array(buf, offset, count);
    offset += count;
  } else {
    chainContinue = new Uint8Array(count);
  }

  let segType: Uint8Array;
  if (flags & 8) {
    segType = new Uint8Array(buf, offset, count);
  } else {
    segType = new Uint8Array(count);
  }

  const zSet = new Set<number>();
  for (let i = 0; i < count; i++) {
    zSet.add(geoms[i * 8 + 7]);
  }
  const layerZs = Float32Array.from(zSet).sort();

  return { count, geoms, roles, chainContinue, segType, layerZs };
}
