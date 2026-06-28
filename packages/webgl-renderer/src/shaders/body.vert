#version 300 es
// body.vert — GLSL ES 3.0
// Instanced body vertex shader: reads segment data from float texture via texelFetch.
// Coordinate mapping: geometry Y → world Z (upDir = (0,0,1)), X → world right, Z → forward.
precision highp float;

// ── Vertex input ──
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in float segIndex; // instanced: which segment this instance renders

// ── Uniforms (group(0) equivalent) ──
layout(std140) uniform Camera {
    mat4 viewProj;
    mat4 viewMat;
    vec3 camPos;
};

layout(std140) uniform Material {
    float roughness;
    float metalness;
    float envIntensity;
    float specularStrength;
    float ambientStrength;
    float useRoleColors;
    // 8 bytes padding before vec3
    vec3 baseColorTint;
};

uniform sampler2D segTex;            // per-segment data (4 texels/segment)
uniform sampler2D colorTex;          // per-segment role colors (1 texel/segment)
uniform int texWidth;                // width of segTex/colorTex in texels

// ── Output ──
out vec3 vWorldPos;
out vec3 vWorldNormal;
out vec3 vColor;
out vec4 vClipPos;   // for shadow noise

// ── Helpers ──
ivec2 texCoord(int idx) {
    return ivec2(idx % texWidth, idx / texWidth);
}

void main() {
    int ii = int(segIndex);

    // ── Read segment data (4 texels) ──
    int t = ii * 4;
    vec4 startPos_ = texelFetch(segTex, texCoord(t),     0);
    vec4 endPos_   = texelFetch(segTex, texCoord(t + 1), 0);
    vec4 chain_    = texelFetch(segTex, texCoord(t + 2), 0);
    vec4 misc_     = texelFetch(segTex, texCoord(t + 3), 0);

    vec3 segColor = texelFetch(colorTex, texCoord(ii), 0).rgb;

    float tParam = position.z + 0.5;
    uint packed = uint(misc_.x);
    bool isArc = (packed & 1u) != 0u;
    float width = startPos_.w;
    // Cull zero-width (hidden) segments
    if (width <= 0.0) {
        gl_Position = vec4(0.0);
        return;
    }

    vec3 segPos;
    vec3 endTangent;

    if (isArc) {
        // Rational quadratic Bézier: p0 = start, p1 = end, p2 = next segment start
        vec3 p0 = startPos_.xyz;
        vec3 p1 = endPos_.xyz;
        vec3 p2 = texelFetch(segTex, texCoord((ii + 1) * 4), 0).xyz;
        float w = endPos_.w;
        float mt = 1.0 - tParam;
        float mt2 = mt * mt;
        float t2 = tParam * tParam;
        float denom = mt2 + 2.0 * tParam * mt * w + t2;
        segPos = (mt2 * p0 + 2.0 * tParam * mt * w * p1 + t2 * p2) / denom;

        // Finite-difference tangent
        float eps = 0.01;
        float te = min(tParam + eps, 1.0);
        float me = 1.0 - te;
        float me2 = me * me;
        float te2 = te * te;
        float de = me2 + 2.0 * te * me * w + te2;
        vec3 pe = (me2 * p0 + 2.0 * te * me * w * p1 + te2 * p2) / de;
        float ts = max(tParam - eps, 0.0);
        float ms = 1.0 - ts;
        float ms2 = ms * ms;
        float ts2 = ts * ts;
        float ds = ms2 + 2.0 * ts * ms * w + ts2;
        vec3 ps = (ms2 * p0 + 2.0 * ts * ms * w * p1 + ts2 * p2) / ds;
        vec3 dDir = pe - ps;
        endTangent = length(dDir) < 0.0001 ? vec3(0.0, 0.0, 1.0) : normalize(dDir);
    } else {
        segPos = mix(startPos_.xyz, endPos_.xyz, tParam);
        vec3 dir = endPos_.xyz - startPos_.xyz;
        float segLen = length(dir);
        endTangent = segLen < 0.001 ? vec3(0.0, 0.0, 1.0) : dir / segLen;
    }

    // ── Interpolate between chain-start and current tangent ──
    vec3 chainStartTangent = chain_.xyz;
    vec3 tangent;
    float cstLen = length(chainStartTangent);
    if (isArc) {
        tangent = endTangent;
    } else if (cstLen > 0.001) {
        tangent = normalize(mix(chainStartTangent, endTangent, tParam));
    } else {
        tangent = endTangent;
    }

    // ── Orthonormal basis (Z-up) ──
    vec3 upDir = vec3(0.0, 0.0, 1.0);
    vec3 rightDir = -normalize(cross(upDir, tangent));
    if (length(rightDir) < 0.001) {
        rightDir = vec3(1.0, 0.0, 0.0);
    }
    vec3 fwdDir = -normalize(cross(rightDir, upDir));
    mat3 rot = mat3(rightDir, upDir, fwdDir);

    float hScale = 1.25;
    float areaCorrection = 1.1;
    vec3 local = vec3(
        position.x * width * areaCorrection,
        position.y * width * hScale,
        0.0
    );

    vec3 worldPos = segPos + rot * local;
    vec3 worldNormal = normalize(rot * normal);

    gl_Position = viewProj * vec4(worldPos, 1.0);
    vWorldPos = worldPos;
    vWorldNormal = worldNormal;
    vColor = mix(baseColorTint, segColor * baseColorTint, useRoleColors);
    vClipPos = gl_Position;
}
