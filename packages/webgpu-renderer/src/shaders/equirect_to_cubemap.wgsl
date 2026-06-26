// ── Equirectangular HDR → Cubemap ──
//
// Purpose:
//   Renders each cubemap face (6 faces × 10 mip levels) by sampling
//   the equirectangular (lat-long) HDR environment map.
//   The cubemap is the starting point for all downstream IBL textures:
//   irradiance map, specular prefilter, and (indirectly) the BRDF LUT.
//
// Coordinate convention:
//   Input (equirect HDR):        Y-up. uv.y=0 = top of image = zenith (sky),
//                                 uv.y=1 = bottom = nadir (ground).
//                                 uv.x=0 = left seam, uv.x=1 = right seam.
//   Output (cubemap faces):      Built using the standard WebGPU/Vulkan cubemap
//                                 convention (right=(1,0,0), up=(0,1,0) for +Z face).
//                                 See makeFaceMatrices() in ibl.ts for the 6 face
//                                 view/projection matrices.
//   Scene (runtime sampling):    Z-up (+Z = sky). The conversion from world-space
//                                 direction dir to equirect UV is:
//                                   phi   = atan2(dir.y, dir.x)  — azimuth from +X toward +Y
//                                   theta = asin(dir.z)          — elevation (Z = up)
//                                   uv.y  = 0.5 - theta / π      — top = zenith
//                                   uv.x  = phi / (2π) + 0.5     — center = +X
//
// Key detail — azimuth convention:
//   Using atan2(dir.z, dir.x) puts the singularity on the ±Y faces (horizontal),
//   causing polar-like distortion in the north/south cubemap faces.
//   Using atan2(dir.y, dir.x) puts the singularity on the ±Z faces
//   (zenith/nadir) where polar distortion is expected and less noticeable.
//   Both place +X at the center of the equirect image (uv.x = 0.5), matching
//   the Poly Haven HDRI convention.
//
// Key detail — Z-up elevation:
//   The scene world has Z as the vertical axis. asin(dir.z) gives the
//   elevation: positive z → above horizon (sky), negative z → below (ground).
//   This is why theta = asin(dir.z), NOT asin(dir.y) (which would assume Y-up).
//
// Azimuth convention:
//   Poly Haven HDRIs use +X as the "forward" direction at the center of the
//   equirect image (uv.x = 0.5). Using atan2(dir.z, dir.x) places +X at the
//   center. Using atan2(dir.x, dir.z) would place +Z at center, rotating the
//   environment 90° horizontally.
//
// Face matrices (defined in ibl.ts makeFaceMatrices()):
//   Each face is rendered with a 90° FOV perspective camera at the origin,
//   looking along the face's principal axis. The vertex positions from
//   CUBE_VERTS are in world XYZ — the same Z-up system. So dir.z correctly
//   captures elevation for all 6 faces.
//
// Pipeline:
//   - Vertex: passes through raw cube vertex as worldPos (no model transform)
//   - Fragment: normalizes worldPos to get the direction, computes UV, samples HDR
//   - Output: rgba16float per face per mip (format = 'rgba16float')
//
// Precision:
//   The source equirect texture is rgba16float (loaded from .hdr file via RGBE decode).
//   The output cubemap is also rgba16float. All IBL textures share this precision,
//   providing ~3.3 decimal digits of dynamic range — sufficient for environment lighting.

@group(0) @binding(0) var<uniform> mvp: mat4x4<f32>;
@group(0) @binding(1) var equirectTex: texture_2d<f32>;
@group(0) @binding(2) var equirectSampler: sampler;

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
};

@vertex
fn vs_main(@location(0) position: vec4<f32>) -> VSOut {
  var out: VSOut;
  out.pos = mvp * position;       // face-specific MVP (90° FOV, face direction)
  out.worldPos = position.xyz;    // raw cube vertex = direction from origin
  return out;
}

@fragment
fn fs_main(@location(0) worldPos: vec3<f32>) -> @location(0) vec4<f32> {
  let dir = normalize(worldPos);

  // Spherical coords — Z-up: dir.z is the vertical component.
  // Azimuth measured from +X toward +Y: atan2(dir.y, dir.x).
  // This puts the singularity on the +Z/-Z faces (zenith/nadir) where polar
  // distortion is expected, keeping horizontal faces (X, Y) rectilinear.
  let phi = atan2(dir.y, dir.x);            // azimuth: 0 = +X, π/2 = +Y
  let theta = asin(dir.z);                  // elevation: π/2 = zenith, -π/2 = nadir

  // Map to equirect UV. uv.x = 0.5 is +X (Poly Haven convention). uv.y = 0 is zenith.
  let uv = vec2<f32>(phi * 0.1591549 + 0.5, 0.5 - theta * 0.3183099);
  //   0.1591549 = 1 / (2π)    — phi ∈ [-π, π] → uv.x ∈ [0, 1]
  //   0.3183099 = 1 / π       — theta ∈ [-π/2, π/2] → uv.y ∈ [0, 1]

  let color = textureSample(equirectTex, equirectSampler, uv).rgb;
  return vec4<f32>(color, 1.0);
}
