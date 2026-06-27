// ── Body vertex shader ──
// Transforms the 2D cross‑section (XY) into world space (Z‑up).
//
// geometry Y  → world Z  (upDir = (0,0,1))
// geometry X  → world right
// geometry Z  → world forward (along segment tangent)

@vertex
fn vs_main(in: VertexInput, @builtin(instance_index) ii: u32) -> VertexOutput {
  // Cull segments not assigned to this LOD
  if (segmentLod[ii] != lodLevel) {
    var out: VertexOutput;
    out.clipPos = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    return out;
  }

  let data = segments[ii];
  let segColor = colors[ii].rgb;

  let t: f32 = in.position.z + 0.5;
  let packed: u32 = u32(data.misc.x);
  let isArc: bool = (packed & 1u) != 0u;
  let width: f32 = data.startPos.w;

  // Evaluate segment position and tangent
  var segPos: vec3<f32>;
  var endTangent: vec3<f32>;

  if (isArc) {
    // Rational quadratic Bézier
    let p0 = data.startPos.xyz;
    let p1 = data.endPos.xyz;
    let p2 = segments[ii + 1u].startPos.xyz;
    let w  = data.endPos.w;
    let mt = 1.0 - t;
    let mt2 = mt * mt;
    let t2 = t * t;
    let denom = mt2 + 2.0 * t * mt * w + t2;
    segPos = (mt2 * p0 + 2.0 * t * mt * w * p1 + t2 * p2) / denom;

    // Finite‑difference tangent
    let eps = 0.01;
    let te = min(t + eps, 1.0); let me = 1.0 - te;
    let me2 = me * me; let te2 = te * te;
    let de = me2 + 2.0 * te * me * w + te2;
    let pe = (me2 * p0 + 2.0 * te * me * w * p1 + te2 * p2) / de;
    let ts = max(t - eps, 0.0); let ms = 1.0 - ts;
    let ms2 = ms * ms; let ts2 = ts * ts;
    let ds = ms2 + 2.0 * ts * ms * w + ts2;
    let ps = (ms2 * p0 + 2.0 * ts * ms * w * p1 + ts2 * p2) / ds;
    let dDir = pe - ps;
    endTangent = select(normalize(dDir), vec3<f32>(0.0, 0.0, 1.0), length(dDir) < 0.0001);
  } else {
    segPos = mix(data.startPos.xyz, data.endPos.xyz, t);
    let dir = data.endPos.xyz - data.startPos.xyz;
    let segLen = length(dir);
    endTangent = select(dir / segLen, vec3<f32>(0.0, 0.0, 1.0), segLen < 0.001);
  }

  // Interpolate between chain‑start tangent and current endTangent
  let chainStartTangent = data.chain.xyz;
  var tangent: vec3<f32>;
  let cstLen = length(chainStartTangent);
  if (isArc) {
    tangent = endTangent;
  } else if (cstLen > 0.001) {
    tangent = normalize(mix(chainStartTangent, endTangent, t));
  } else {
    tangent = endTangent;
  }

  // Build orthonormal basis: upDir = (0,0,1), right & forward from tangent
  let upDir = vec3<f32>(0.0, 0.0, 1.0);
  var rightDir = -normalize(cross(upDir, tangent));
  if (length(rightDir) < 0.001) {
    rightDir = vec3<f32>(1.0, 0.0, 0.0);
  }
  let fwdDir = -normalize(cross(rightDir, upDir));
  let rot = mat3x3<f32>(rightDir, upDir, fwdDir);

  let hScale = 1.25;
  let areaCorrection = 1.1;
  // Local position: map geometry XY to right/up, Z is unused (zero)
  let local = vec3<f32>(
    in.position.x * width * areaCorrection,
    in.position.y * width * hScale,
    0.0,
  );

  let worldPos = segPos + rot * local;

  // Use the pre‑computed 2D outward normal directly.
  // No blending – it already points correctly everywhere (flat top, curved sides, bottom).
  let packedFlags = u32(data.misc.x);
  let worldNormal = normalize(rot * (in.normal));

  var out: VertexOutput;
  out.clipPos = camera.viewProj * vec4<f32>(worldPos, 1.0);
  out.worldPos = worldPos;
  out.worldNormal = worldNormal;
  out.color = mix(material.baseColorTint, segColor * material.baseColorTint, material.useRoleColors);
  return out;
}
