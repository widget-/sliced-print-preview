// cap.wgsl
// ── Cap (rounded end) vertex shader ──
//
// Same coordinate mapping as segment.wgsl:
//   geometry Y  → world Z  (upDir = (0,0,1))
//   geometry X  → world right direction
//   geometry Z  → world forward (along segment tangent)
// The cap dome (bulge) faces outward along the segment tangent.

@vertex
fn vs_cap(in: VertexInput, @builtin(instance_index) ii: u32) -> VertexOutput {
  let capInfo: vec2<f32> = capInstances[ii];
  let segIdx: u32 = u32(capInfo.x);
  let isEnd: f32 = capInfo.y;

  // Cull caps on LOD 2 segments
  if (segmentLod[segIdx] >= 2u) {
    var out: VertexOutput;
    out.clipPos = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    return out;
  }

  let data = segments[segIdx];
  let segColor = colors[segIdx].rgb;
  let capsWidth: f32 = data.startPos.w;

  // Position at start or end of segment
  let pos: vec3<f32> = select(data.startPos.xyz, data.endPos.xyz, isEnd > 0.5);

  // Tangent direction – always outward from the endpoint
  let dir: vec3<f32> = data.endPos.xyz - data.startPos.xyz;
  let segLen: f32 = length(dir);
  var tangent: vec3<f32>;
  if (segLen > 0.001) {
    tangent = dir / segLen;
  } else {
    tangent = vec3<f32>(0.0, 0.0, 1.0);
  }

  // Build orthonormal basis
  let upDir: vec3<f32> = vec3<f32>(0.0, 0.0, 1.0);
  var rightDir: vec3<f32> = -normalize(cross(upDir, tangent));
  if (length(rightDir) < 0.001) {
    rightDir = vec3<f32>(1.0, 0.0, 0.0);
  }
  let fwdDir: vec3<f32> = -normalize(cross(rightDir, upDir));
  let rot = mat3x3<f32>(rightDir, upDir, fwdDir);

  let hScale: f32 = 1.25;
  let areaCorrection: f32 = 1.1;

  // Flip X and Z for an end cap so the bulge faces the correct way.
  // flipEnd =  1 → end cap (normal direction unchanged, bulge along +Z)
  // flipEnd = -1 → start cap (bulge along -Z)
  let flipEnd: f32 = select(-1.0, 1.0, isEnd > 0.5);

  let local: vec3<f32> = vec3<f32>(
    flipEnd * in.position.x * capsWidth * areaCorrection,
            in.position.y * capsWidth * hScale,
    flipEnd * in.position.z * capsWidth * 0.5,
  );

  let worldPos: vec3<f32> = pos + rot * local;

  // The dome geometry's stored normal always bulges in +Z (the geometry is
  // built with normals that have a positive Z component). For a start cap
  // we must reverse that Z so the normal points opposite fwdDir.
  let localNormal: vec3<f32> = vec3<f32>(
    flipEnd * in.normal.x,
    in.normal.y,
    flipEnd * in.normal.z,
  );
  let worldNormal: vec3<f32> = normalize(rot * localNormal);

  var out: VertexOutput;
  out.clipPos = camera.viewProj * vec4<f32>(worldPos, 1.0);
  out.worldPos = worldPos;
  out.worldNormal = worldNormal;
  out.color = mix(material.baseColorTint, segColor * material.baseColorTint, material.useRoleColors);
  return out;
}
