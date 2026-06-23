// ── Babylon.js ShaderMaterial shader source (ESSL 3.0 / WebGL2) ────────────

export const SEGMENT_VERTEX_SHADER = /* glsl */ `
  precision highp float;

  in vec3 position;
  in vec3 normal;
  in vec4 instColor;
  in vec4 instVisible;

  uniform mat4 worldViewProjection;
  uniform mat4 view;

  uniform sampler2D segmentTexture;
  uniform float uSegmentCount;
  uniform float uHeightScale;
  uniform float uAreaCorrection;
  uniform vec2 uTexSize;
  uniform float uDbgNormFlip;

  out vec3 vNormal;
  out vec3 vWorldPos;
  out float vLayerZ;
  out vec3 vInstanceColor;
  out vec3 vTangent;

  ivec2 texelCoord(int id, int row) {
    int idx = id * 4 + row;
    int w = int(uTexSize.x);
    return ivec2(idx % w, idx / w);
  }

  void main() {
    vec4 row0 = texelFetch(segmentTexture, texelCoord(gl_InstanceID, 0), 0);
    vec4 row1 = texelFetch(segmentTexture, texelCoord(gl_InstanceID, 1), 0);
    vec4 row2 = texelFetch(segmentTexture, texelCoord(gl_InstanceID, 2), 0);
    vec4 row3 = texelFetch(segmentTexture, texelCoord(gl_InstanceID, 3), 0);

    float t = position.z + 0.5;

    vec3 segPos;
    vec3 endTangent;
    float layerZ;
    float width;

    if (row2.z > 0.5) {
      vec3 p0 = row0.xyz;
      width    = row0.w;
      vec3 p1 = row1.xyz;
      float w = row1.w;
      layerZ   = row2.w;
      vec4 nextRow0 = texelFetch(segmentTexture, texelCoord(gl_InstanceID + 1, 0), 0);
      vec3 p2 = nextRow0.xyz;

      float mt = 1.0 - t;
      float denom = mt*mt + 2.0*t*mt*w + t*t;
      segPos = (mt*mt*p0 + 2.0*t*mt*w*p1 + t*t*p2) / denom;

      float eps = 0.001;
      float te = min(t + eps, 1.0); float me = 1.0 - te;
      float de = me*me + 2.0*te*me*w + te*te;
      vec3 pe = (me*me*p0 + 2.0*te*me*w*p1 + te*te*p2) / de;
      float ts = max(t - eps, 0.0); float ms = 1.0 - ts;
      float ds = ms*ms + 2.0*ts*ms*w + ts*ts;
      vec3 ps = (ms*ms*p0 + 2.0*ts*ms*w*p1 + ts*ts*p2) / ds;
      endTangent = normalize(pe - ps);
    } else {
      vec3 start = row0.xyz;
      layerZ     = row0.w;
      vec3 end   = row1.xyz;
      width      = row1.w;
      segPos = mix(start, end, t);
      vec3 dir = end - start;
      float len = length(dir);
      endTangent = len > 0.001 ? dir / len : vec3(0.0, 0.0, 1.0);
    }

    vec3 chainStartTangent = row3.xyz;
    vec3 tangent = normalize(mix(chainStartTangent, endTangent, t));

    vec3 upDir   = vec3(0.0, 0.0, 1.0);
    vec3 rightDir = -normalize(cross(upDir, tangent));
    if (length(rightDir) < 0.001) {
      rightDir = vec3(1.0, 0.0, 0.0);
    }
    vec3 fwdDir = -normalize(cross(rightDir, upDir));
    mat3 rot   = mat3(rightDir, upDir, fwdDir);

    vec3 local = vec3(position.x * width * uAreaCorrection, position.y * width * uHeightScale, 0.0);
    vec3 worldPos = segPos + rot * local;

    vNormal = normalize(rot * normal);

    // Debug: flip bottom normals by bitmask (bits 0=left, 1=right)
    if (uDbgNormFlip > 0.5 && position.y < -0.17) {
      float m = uDbgNormFlip;
      bool flip = position.x < 0.0 ? mod(m, 2.0) > 0.5 : mod(floor(m / 2.0), 2.0) > 0.5;
      if (flip) vNormal = -vNormal;
    }
    vWorldPos = worldPos;
    vLayerZ = layerZ;
    vInstanceColor = instColor.rgb;
    vTangent = tangent;

    gl_Position = worldViewProjection * vec4(worldPos, 1.0);
    if (instVisible.x < 0.5) gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
  }
`;

export const ENDCAP_VERTEX_SHADER = /* glsl */ `
  precision highp float;

  in vec3 position;
  in vec3 normal;
  in vec4 instColor;
  in vec4 instVisible;

  uniform mat4 worldViewProjection;
  uniform mat4 view;

  uniform sampler2D segmentTexture;
  uniform sampler2D capMapTexture;
  uniform float uSegmentCount;
  uniform float uHeightScale;
  uniform float uAreaCorrection;
  uniform vec2 uTexSize;
  uniform vec2 uCapTexSize;

  out vec3 vNormal;
  out vec3 vWorldPos;
  out float vLayerZ;
  out vec3 vInstanceColor;
  out vec3 vTangent;

  ivec2 texelCoord(int id, int row, float w) {
    int idx = id * 4 + row;
    return ivec2(idx % int(w), idx / int(w));
  }

  ivec2 capTexelCoord(int id, float w) {
    return ivec2(id % int(w), id / int(w));
  }

  void main() {
    int capIdx = gl_InstanceID;
    vec4 capRow = texelFetch(capMapTexture, capTexelCoord(capIdx, uCapTexSize.x), 0);
    int segIdx = int(capRow.r);
    int isEnd  = int(capRow.g);

    vec4 row0 = texelFetch(segmentTexture, texelCoord(segIdx, 0, uTexSize.x), 0);
    vec4 row1 = texelFetch(segmentTexture, texelCoord(segIdx, 1, uTexSize.x), 0);
    vec4 row2 = texelFetch(segmentTexture, texelCoord(segIdx, 2, uTexSize.x), 0);

    float width;
    float layerZ;
    vec3 endpoint;
    vec3 tangent;

    if (row2.z > 0.5) {
      width  = row0.w;
      layerZ = row2.w;
      vec3 p0 = row0.xyz;
      vec3 p1 = row1.xyz;

      if (isEnd == 0) {
        endpoint = p0;
        vec3 dir = p1 - p0;
        float dlen = length(dir);
        tangent = dlen > 0.001 ? dir / dlen : vec3(0.0, 0.0, 1.0);
      } else {
        vec4 nextRow0 = texelFetch(segmentTexture, texelCoord(segIdx + 1, 0, uTexSize.x), 0);
        vec3 p2 = nextRow0.xyz;
        endpoint = p2;
        vec3 dir = p2 - p1;
        float dlen = length(dir);
        tangent = dlen > 0.001 ? dir / dlen : vec3(0.0, 0.0, 1.0);
      }
    } else {
      vec3 start = row0.xyz;
      layerZ     = row0.w;
      vec3 end   = row1.xyz;
      width      = row1.w;

      endpoint = isEnd == 0 ? start : end;

      vec3 dir = end - start;
      float dlen = length(dir);
      tangent = dlen > 0.001 ? dir / dlen : vec3(0.0, 0.0, 1.0);
    }

    float sign = isEnd == 0 ? -1.0 : 1.0;

    vec3 upDir   = vec3(0.0, 0.0, 1.0);
    vec3 rightDir = -normalize(cross(upDir, tangent));
    if (length(rightDir) < 0.001) {
      rightDir = vec3(1.0, 0.0, 0.0);
    }
    vec3 fwdDir = -normalize(cross(rightDir, upDir));
    mat3 rot   = mat3(rightDir, upDir, fwdDir);

    float capR = width * 0.5;

    vec3 local = vec3(
      position.x * width * uAreaCorrection,
      position.y * width * uHeightScale,
      position.z * capR * sign);
    vec3 worldPos = endpoint + rot * local;

    vec3 N = vec3(normal.x, normal.y, normal.z * sign);
    vNormal = normalize(rot * N);

    vWorldPos = worldPos;
    vLayerZ = layerZ;
    vInstanceColor = instColor.rgb;
    vTangent = tangent;

    gl_Position = worldViewProjection * vec4(worldPos, 1.0);
    if (instVisible.x < 0.5) gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
  }
`;

export const SEGMENT_FRAGMENT_SHADER = /* glsl */ `
  precision highp float;

  uniform float uRoughness;
  uniform float uMetalness;
  uniform vec3 uKeyLightDir;
  uniform vec3 uCameraPos;
  uniform sampler2D uShadowMap;
  uniform mat4 uShadowMatrix;
  uniform vec2 uShadowMapSize;
  uniform sampler2D uEnvMapEQ;
  uniform float uEnvMapLOD;
  uniform float uEnvIntensity;
  uniform vec3 uBaseColorTint;
  uniform float uSpecularStrength;
  uniform float uAmbientStrength;
  uniform float uLodMode; // 0=normal, 1=cylinder-integrated
  uniform float uDbgHighlight; // 0=normal, 1=cap highlight
  uniform float uDbgNormVis;   // 0=normal, 1=show N as color

  in vec3 vNormal;
  in vec3 vWorldPos;
  in float vLayerZ;
  in vec3 vInstanceColor;
  in vec3 vTangent;

  layout(location = 0) out vec4 fragColor;

  const float PI = 3.14159265359;

  float D_GGX(float NdotH, float a2) {
    float d = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
  }

  float G1_Smith(float NdotX, float alpha2) {
    float k = alpha2 / 2.0;
    return NdotX / (NdotX * (1.0 - k) + k);
  }

  float G_Smith(float NdotV, float NdotL, float alpha2) {
    return G1_Smith(NdotV, alpha2) * G1_Smith(NdotL, alpha2);
  }

  vec3 F_Schlick(vec3 F0, float VdotH) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0);
  }

  vec3 F_SchlickRoughness(vec3 F0, float VdotH, float roughness) {
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0);
  }

  const vec2 poissonDisk[16] = vec2[16](
    vec2(-0.94201624, -0.39906216), vec2( 0.94558609, -0.76890725),
    vec2(-0.09418410, -0.92938870), vec2( 0.34495938,  0.29387760),
    vec2(-0.91588581,  0.45771432), vec2(-0.81544232, -0.87912464),
    vec2(-0.38277543,  0.27676845), vec2( 0.97484398,  0.75648379),
    vec2( 0.44323325, -0.97511554), vec2( 0.53742981, -0.47373420),
    vec2(-0.26496911, -0.41893023), vec2( 0.79197514,  0.19090188),
    vec2(-0.24188840,  0.99706507), vec2(-0.81409955,  0.91437590),
    vec2( 0.19984126,  0.78641367), vec2( 0.14383161, -0.14100790)
  );

  float shadowFactor(vec3 worldPos, vec3 N, vec3 L) {
    vec4 p = uShadowMatrix * vec4(worldPos, 1.0);
    p.xyz /= p.w;
    p.xyz = p.xyz * 0.5 + 0.5;
    if (p.x < 0.0 || p.x > 1.0 || p.y < 0.0 || p.y > 1.0 || p.z > 1.0 || p.z < 0.0) return 1.0;
    float bias = max(0.0015, 0.003 * (1.0 - dot(N, L)));
    vec2 ts = 1.0 / uShadowMapSize;
    float sum = 0.0;
    for (int i = 0; i < 16; i++) {
      vec2 uv = p.xy + poissonDisk[i] * ts * 2.5;
      float d = texture(uShadowMap, uv).r;
      sum += (p.z - bias > d) ? 0.0 : 1.0;
    }
    return sum / 16.0;
  }

  vec2 dirToEquirect(vec3 dir) {
    float u = atan(dir.y, dir.x) / (2.0 * PI) + 0.5;
    float v = asin(clamp(dir.z, -1.0, 1.0)) / PI + 0.5;
    return vec2(u, v);
  }

  vec3 sampleEnvEQ(sampler2D tex, vec3 dir, float lod) {
    return textureLod(tex, dirToEquirect(dir), lod).rgb;
  }

  // Oren-Nayar diffuse for rough surfaces (scatters light, softens gradients)
  float orenNayarFactor(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.001);

    float sigmaSq = roughness * roughness;
    float A = 1.0 - 0.5 * sigmaSq / (sigmaSq + 0.33);
    float B = 0.45 * sigmaSq / (sigmaSq + 0.09);

    float sin_alpha = sqrt(max(0.0, 1.0 - max(NdotL, NdotV) * max(NdotL, NdotV)));
    float tan_beta  = sqrt(max(0.0, 1.0 - min(NdotL, NdotV) * min(NdotL, NdotV)))
                    / max(min(NdotL, NdotV), 0.001);

    vec3 L_proj = normalize(L - N * NdotL);
    vec3 V_proj = normalize(V - N * NdotV);
    float cos_phi_diff = max(dot(L_proj, V_proj), 0.0);

    return mix(A + B * cos_phi_diff * sin_alpha * tan_beta, 1.0, roughness);
  }

  // Cylinder-integrated diffuse: average of max(0, N·L) over all normals ⟂ T
  float cylDiffuse(vec3 T, vec3 L) {
    float tdl = dot(T, normalize(L));
    return 0.63662 * sqrt(max(0.0, 1.0 - tdl * tdl));
  }

  void main() {
    vec3 N = normalize(vNormal);
    vec3 V = normalize(uCameraPos - vWorldPos);

    float roughness = max(uRoughness, 0.04);
    float alpha2 = roughness * roughness;

    vec3 baseColor = vInstanceColor * uBaseColorTint;
    vec3 F0 = mix(vec3(0.04), baseColor, uMetalness);
    vec3 diffuseColor = baseColor * (1.0 - uMetalness);
    float NdotV = max(dot(N, V), 0.001);

    vec3 Lo = vec3(0.0);

    if (uLodMode > 0.5) {
      // ── LOD 2: replace normal with view-facing cylinder normal ──
      // The flat quad has degenerate interpolated normals, but the
      // tangent is correct. Use the view-facing cylinder normal to
      // keep correct material response (specular, roughness, Fresnel).
      vec3 T = normalize(vTangent);
      N = normalize(cross(cross(T, V), T));
    }

    // ── Normal per-pixel lighting (LOD 0 / LOD 1 / LOD 2) ───────────
    {
        vec3 L = normalize(uKeyLightDir);
      vec3 H = normalize(V + L);
      float NdotL = max(dot(N, L), 0.0);
      float NdotH = max(dot(N, H), 0.0);
      float VdotH = max(dot(V, H), 0.0);
      float D = D_GGX(NdotH, alpha2);
      float G = G_Smith(NdotV, NdotL, alpha2);
      vec3  F = F_Schlick(F0, VdotH) * uSpecularStrength;
      vec3 spec = D * G * F / max(4.0 * NdotL * NdotV, 0.001);
      vec3 kD = (1.0 - F) * (1.0 - uMetalness);
      float sf = shadowFactor(vWorldPos, N, L);
      vec3 Li = vec3(1.0);
      Lo += (kD * diffuseColor / PI * orenNayarFactor(N, V, L, roughness) + spec) * Li * NdotL * sf;
    }

    {
      vec3 L = normalize(vec3(-0.5, 0.25, -0.3));
      vec3 H = normalize(V + L);
      float NdotL = max(dot(N, L), 0.0);
      float NdotH = max(dot(N, H), 0.0);
      float VdotH = max(dot(V, H), 0.0);
      float D = D_GGX(NdotH, alpha2);
      float G = G_Smith(NdotV, NdotL, alpha2);
      vec3  F = F_Schlick(F0, VdotH) * uSpecularStrength;
      vec3 spec = D * G * F / max(4.0 * NdotL * NdotV, 0.001);
      vec3 kD = (1.0 - F) * (1.0 - uMetalness);
      vec3 Li = vec3(0.6, 0.55, 0.45);
      Lo += (kD * diffuseColor / PI * orenNayarFactor(N, V, L, roughness) + spec) * Li * NdotL;
    }

    vec3 ambient = vec3(0.0);
    vec3 R = reflect(-V, N);

    // Env map contribution (controlled by envIntensity)
    if (uEnvMapLOD > 0.0) {
      // Fade env for downward-facing surfaces to avoid sky bleed on bottom
      float horizon = smoothstep(-0.3, 0.0, dot(N, vec3(0.0, 0.0, 1.0)));
      float diffuseLOD = uEnvMapLOD * 0.85;
      vec3 irradiance = sampleEnvEQ(uEnvMapEQ, N, diffuseLOD) * uEnvIntensity * horizon;
      vec3 F_ibl = F_SchlickRoughness(F0, NdotV, roughness) * uSpecularStrength;
      vec3 kD_ibl = (1.0 - F_ibl) * (1.0 - uMetalness);
      ambient += kD_ibl * diffuseColor * irradiance;
      float specLOD = roughness * uEnvMapLOD;
      vec3 prefiltered = sampleEnvEQ(uEnvMapEQ, R, specLOD) * uEnvIntensity * horizon;
      ambient += prefiltered * F_ibl;
    }

    // Hemisphere ambient fill (always on, controlled by ambientStrength)
    {
      float NdotUp = dot(N, vec3(0.0, 0.0, 1.0));
      vec3 skyColor = vec3(0.6, 0.6, 0.7);
      vec3 groundColor = vec3(0.3, 0.25, 0.2);
      ambient += uAmbientStrength * mix(groundColor, skyColor, NdotUp * 0.5 + 0.5) * diffuseColor;
    }

    float ao = 0.80 + 0.20 * smoothstep(0.0, 30.0, vLayerZ);
    vec3 final = (Lo + ambient) * ao;
    // Reinhard tone mapping + gamma correction
    final = final / (final + vec3(1.0));
    final = pow(final, vec3(1.0 / 2.2));
    fragColor = vec4(final, 1.0);
  }
`;

export const DEPTH_FRAGMENT_SHADER = /* glsl */ `
  in vec3 vNormal;
  in vec3 vWorldPos;
  in float vLayerZ;
  in vec3 vInstanceColor;

  uniform float uCameraFar;

  layout(location = 0) out vec4 fragColor;

  void main() {
    float ndc = gl_FragCoord.z * 2.0 - 1.0;
    float linear = (2.0 * 0.1 * uCameraFar) / (uCameraFar + 0.1 - ndc * (uCameraFar - 0.1));
    fragColor = vec4(linear / uCameraFar, 0.0, 0.0, 0.0);
  }
`;

