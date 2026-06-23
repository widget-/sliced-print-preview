@vertex
fn vs_main(in: VertexInput, @builtin(instance_index) ii: u32) -> VertexOutput {
  let data = segments[ii];
  let segColor = colors[ii].rgb;

  let t: f32 = in.position.z + 0.5;
  let packed: u32 = u32(data.misc.x);
  let isArc: bool = (packed & 1u) != 0u;
  let width: f32 = data.startPos.w;

  var segPos: vec3<f32>;
  var endTangent: vec3<f32>;

  if (isArc) {
    // Rational quadratic Bézier: P0 = start, P1 = end (control), P2 = next.start
    let p0: vec3<f32> = data.startPos.xyz;
    let p1: vec3<f32> = data.endPos.xyz;
    let p2: vec3<f32> = segments[ii + 1u].startPos.xyz;
    let w: f32 = data.endPos.w;
    let mt: f32 = 1.0 - t;
    let mt2: f32 = mt * mt;
    let t2: f32 = t * t;
    let denom: f32 = mt2 + 2.0 * t * mt * w + t2;
    segPos = (mt2 * p0 + 2.0 * t * mt * w * p1 + t2 * p2) / denom;

    // Finite-difference tangent for endTangent
    let eps: f32 = 0.01;
    let te: f32 = min(t + eps, 1.0); let me: f32 = 1.0 - te;
    let me2: f32 = me * me; let te2: f32 = te * te;
    let de: f32 = me2 + 2.0 * te * me * w + te2;
    let pe: vec3<f32> = (me2 * p0 + 2.0 * te * me * w * p1 + te2 * p2) / de;
    let ts: f32 = max(t - eps, 0.0); let ms: f32 = 1.0 - ts;
    let ms2: f32 = ms * ms; let ts2: f32 = ts * ts;
    let ds: f32 = ms2 + 2.0 * ts * ms * w + ts2;
    let ps: vec3<f32> = (ms2 * p0 + 2.0 * ts * ms * w * p1 + ts2 * p2) / ds;
    let dDir: vec3<f32> = pe - ps;
    let dLen: f32 = length(dDir);
    endTangent = select(normalize(dDir), vec3<f32>(0.0, 0.0, 1.0), dLen < 0.0001);
  } else {
    segPos = mix(data.startPos.xyz, data.endPos.xyz, t);
    let dir: vec3<f32> = data.endPos.xyz - data.startPos.xyz;
    let segLen: f32 = length(dir);
    endTangent = select(dir / segLen, vec3<f32>(0.0, 0.0, 1.0), segLen < 0.001);
  }

  // Interpolate between chain-start tangent and current endTangent
  let chainStartTangent: vec3<f32> = data.chain.xyz;
  var tangent: vec3<f32>;
  let cstLen: f32 = length(chainStartTangent);
  if (isArc) {
    tangent = endTangent;
  } else if (cstLen > 0.001) {
    tangent = normalize(mix(chainStartTangent, endTangent, t));
  } else {
    tangent = endTangent;
  }

  let upDir: vec3<f32> = vec3<f32>(0.0, 0.0, 1.0);
  var rightDir: vec3<f32> = -normalize(cross(upDir, tangent));
  if (length(rightDir) < 0.001) {
    rightDir = vec3<f32>(1.0, 0.0, 0.0);
  }
  let fwdDir: vec3<f32> = -normalize(cross(rightDir, upDir));
  let rot = mat3x3<f32>(rightDir, upDir, fwdDir);

  let hScale: f32 = 1.25;
  let areaCorrection: f32 = 1.1;
  let local: vec3<f32> = vec3<f32>(
    in.position.x * width * areaCorrection,
    in.position.y * width * hScale,
    0.0
  );

  let worldPos: vec3<f32> = segPos + rot * local;
  let worldNormal: vec3<f32> = rot * in.normal;

  var out: VertexOutput;
  out.clipPos = camera.viewProj * vec4<f32>(worldPos, 1.0);
  out.worldPos = worldPos;
  out.worldNormal = worldNormal;
  out.color = segColor * material.baseColorTint;
  return out;
}
