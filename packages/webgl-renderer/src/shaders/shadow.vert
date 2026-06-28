#version 300 es
// shadow.vert — GLSL ES 3.0
// Depth-only vertex shader for shadow map rendering.
// Same coordinate system as body.vert — reads from segTex.
precision highp float;

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in float segIndex;

uniform mat4 shadowVP;
uniform sampler2D segTex;
uniform int texWidth;

ivec2 texCoord(int idx) {
    return ivec2(idx % texWidth, idx / texWidth);
}

void main() {
    int ii = int(segIndex);

    int t = ii * 4;
    vec4 startPos_ = texelFetch(segTex, texCoord(t),     0);
    vec4 endPos_   = texelFetch(segTex, texCoord(t + 1), 0);
    float width = startPos_.w;
    if (width <= 0.0) {
        gl_Position = vec4(0.0);
        return;
    }

    float tParam = position.z + 0.5;

    vec3 segPos;
    uint packed = uint(texelFetch(segTex, texCoord(t + 3), 0).x);
    bool isArc = (packed & 1u) != 0u;

    if (isArc) {
        vec3 p0 = startPos_.xyz;
        vec3 p1 = endPos_.xyz;
        vec3 p2 = texelFetch(segTex, texCoord((ii + 1) * 4), 0).xyz;
        float w = endPos_.w;
        float mt = 1.0 - tParam;
        float mt2 = mt * mt;
        float t2 = tParam * tParam;
        float denom = mt2 + 2.0 * tParam * mt * w + t2;
        segPos = (mt2 * p0 + 2.0 * tParam * mt * w * p1 + t2 * p2) / denom;
    } else {
        segPos = mix(startPos_.xyz, endPos_.xyz, tParam);
    }

    vec3 tangent;
    if (isArc) {
        tangent = vec3(0.0, 0.0, 1.0);
    } else {
        vec3 dir = endPos_.xyz - startPos_.xyz;
        float segLen = length(dir);
        tangent = segLen < 0.001 ? vec3(0.0, 0.0, 1.0) : dir / segLen;
    }

    vec3 upDir = vec3(0.0, 0.0, 1.0);
    vec3 rightDir = -normalize(cross(upDir, tangent));
    if (length(rightDir) < 0.001) rightDir = vec3(1.0, 0.0, 0.0);
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
    gl_Position = shadowVP * vec4(worldPos, 1.0);
}
