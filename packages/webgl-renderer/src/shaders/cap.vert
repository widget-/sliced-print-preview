#version 300 es
// cap.vert — GLSL ES 3.0
// Instanced endcap vertex shader. Reads cap instance buffer for segment index + isEnd flag.
precision highp float;

// ── Vertex input ──
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in float capSegIndex;   // which segment this cap belongs to
layout(location = 3) in float capIsEnd;       // 0.0 = start, 1.0 = end

// ── Uniforms ──
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
    vec3 baseColorTint;
};

uniform sampler2D segTex;
uniform sampler2D colorTex;
uniform int texWidth;

// ── Output ──
out vec3 vWorldPos;
out vec3 vWorldNormal;
out vec3 vColor;
out vec4 vClipPos;

// ── Helpers ──
ivec2 texCoord(int idx) {
    return ivec2(idx % texWidth, idx / texWidth);
}

void main() {
    int segIdx = int(capSegIndex);
    float isEnd = capIsEnd;

    // ── Read segment data ──
    int t = segIdx * 4;
    vec4 startPos_ = texelFetch(segTex, texCoord(t),     0);
    vec4 endPos_   = texelFetch(segTex, texCoord(t + 1), 0);

    vec3 segColor = texelFetch(colorTex, texCoord(segIdx), 0).rgb;
    float capsWidth = startPos_.w;
    if (capsWidth <= 0.0) {
        gl_Position = vec4(0.0);
        return;
    }

    // ── Position at start or end of segment ──
    vec3 pos = isEnd > 0.5 ? endPos_.xyz : startPos_.xyz;

    // ── Tangent ──
    vec3 dir = endPos_.xyz - startPos_.xyz;
    float segLen = length(dir);
    vec3 tangent = segLen > 0.001 ? dir / segLen : vec3(0.0, 0.0, 1.0);

    // ── Orthonormal basis ──
    vec3 upDir = vec3(0.0, 0.0, 1.0);
    vec3 rightDir = -normalize(cross(upDir, tangent));
    if (length(rightDir) < 0.001) {
        rightDir = vec3(1.0, 0.0, 0.0);
    }
    vec3 fwdDir = -normalize(cross(rightDir, upDir));
    mat3 rot = mat3(rightDir, upDir, fwdDir);

    float hScale = 1.25;
    float areaCorrection = 1.1;

    // flipEnd =  1.0 → end cap (bulge along +Z)
    // flipEnd = -1.0 → start cap (bulge along -Z)
    float flipEnd = isEnd > 0.5 ? 1.0 : -1.0;

    vec3 local = vec3(
        flipEnd * position.x * capsWidth * areaCorrection,
                 position.y * capsWidth * hScale,
        flipEnd * position.z * capsWidth * 0.5
    );

    vec3 worldPos = pos + rot * local;

    vec3 localNormal = vec3(
        flipEnd * normal.x,
        normal.y,
        flipEnd * normal.z
    );
    vec3 worldNormal = normalize(rot * localNormal);

    gl_Position = viewProj * vec4(worldPos, 1.0);
    vWorldPos = worldPos;
    vWorldNormal = worldNormal;
    vColor = mix(baseColorTint, segColor * baseColorTint, useRoleColors);
    vClipPos = gl_Position;
}
