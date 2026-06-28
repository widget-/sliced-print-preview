#version 300 es
// ground.frag — GLSL ES 3.0
// Infinite ground plane with dual-light shadows.

precision highp float;
precision highp sampler2DShadow;

in vec3 vWorldPos;
in vec4 vClipPos;

layout(location = 0) out vec4 outColor;

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

uniform vec4 lightDir;
uniform vec4 lightDir2;

uniform sampler2DShadow shadowTex;
uniform mat4 shadowVP;
uniform float shadowSoftness;

uniform sampler2DShadow shadowTex2;
uniform mat4 shadowVP2;

const vec3 groundColor = vec3(0.35, 0.33, 0.30);

// PCF shadow for ground (8-tap Poisson disk)
float computeGroundShadow(sampler2DShadow tex, mat4 vp, vec3 worldPos) {
    vec4 shadowClip = vp * vec4(worldPos, 1.0);
    vec3 shadowNDC = shadowClip.xyz / shadowClip.w;
    vec2 shadowUV = shadowNDC.xy * 0.5 + 0.5;

    if (shadowUV != clamp(shadowUV, vec2(0.0), vec2(1.0)) ||
        shadowNDC.z < 0.0 || shadowNDC.z > 1.0) {
        return 1.0;
    }

    float texelSize = 1.0 / 1024.0;
    float radius = texelSize * shadowSoftness;
    float phi = fract(sin(dot(vClipPos.xy, vec2(12.9898, 78.233))) * 43758.5453) * 6.283185307;

    float sum = 0.0;
    vec2 offsets[8];
    offsets[0] = vec2( 0.3109,  0.2007);
    offsets[1] = vec2(-0.2234, -0.3367);
    offsets[2] = vec2( 0.4201, -0.1889);
    offsets[3] = vec2(-0.4063,  0.2191);
    offsets[4] = vec2( 0.1298,  0.4862);
    offsets[5] = vec2(-0.4891, -0.0892);
    offsets[6] = vec2(-0.0943, -0.4918);
    offsets[7] = vec2( 0.5013, -0.0247);

    for (int i = 0; i < 8; i++) {
        vec2 off = offsets[i] * radius;
        float s = sin(phi);
        float c = cos(phi);
        vec2 rotOff = vec2(off.x * c - off.y * s, off.x * s + off.y * c);
        sum += texture(tex, vec3(shadowUV + rotOff, shadowNDC.z));
    }
    return sum / 8.0;
}

void main() {
    vec3 upDir = vec3(0.0, 0.0, 1.0);

    float shadowVis1 = computeGroundShadow(shadowTex, shadowVP, vWorldPos);
    float shadowVis2 = computeGroundShadow(shadowTex2, shadowVP2, vWorldPos);

    float ambientFill = max(ambientStrength, 0.05);
    vec3 L1 = normalize(lightDir.xyz);
    vec3 L2 = normalize(lightDir2.xyz);
    float direct1 = max(dot(upDir, L1), 0.0) * lightDir.w;
    float direct2 = max(dot(upDir, L2), 0.0) * lightDir2.w;

    vec3 lit = groundColor * (ambientFill + shadowVis1 * direct1 + shadowVis2 * direct2);

    // Reinhard + gamma
    lit = lit / (lit + 1.0);
    lit = pow(lit, vec3(1.0 / 2.2));

    outColor = vec4(lit, 1.0);
}
