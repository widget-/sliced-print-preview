#version 300 es
// pbr.frag — GLSL ES 3.0
// Cook-Torrance GGX PBR with single shadow map + equirectangular environment map.
// Simplified from pbr.wgsl — single light, no IBL cubemaps.
precision highp float;
precision highp sampler2DShadow;

// ── Inputs ──
in vec3 vWorldPos;
in vec3 vWorldNormal;
in vec3 vColor;
in vec4 vClipPos;

// ── Output ──
layout(location = 0) out vec4 outColor;

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

uniform vec4 lightDir;               // xyz=direction, w=intensity

// Shadow
uniform sampler2DShadow shadowTex;
uniform mat4 shadowVP;
uniform float shadowSoftness;         // PCF kernel radius in texels

// Environment map (equirectangular)
uniform sampler2D envTex;

// ── Constants ──
const float PI = 3.14159265359;
const int   PCF_SAMPLES = 8;

// ── PCF shadow (simple Poisson disk) ──
float computeShadow(vec3 worldPos) {
    vec4 shadowClip = shadowVP * vec4(worldPos, 1.0);
    vec3 shadowNDC = shadowClip.xyz / shadowClip.w;
    vec2 shadowUV = shadowNDC.xy * vec2(0.5, -0.5) + 0.5;

    if (shadowUV != clamp(shadowUV, vec2(0.0), vec2(1.0)) ||
        shadowNDC.z < 0.0 || shadowNDC.z > 1.0) {
        return 1.0;
    }

    float texelSize = 1.0 / 1024.0;
    float radius = texelSize * shadowSoftness;

    // Stochastic rotation from screen position
    float phi = fract(sin(dot(vClipPos.xy, vec2(12.9898, 78.233))) * 43758.5453) * 6.283185307;

    float sum = 0.0;
    // Simple Poisson-like disk (8 samples)
    vec2 offsets[8];
    offsets[0] = vec2( 0.3109,  0.2007);
    offsets[1] = vec2(-0.2234, -0.3367);
    offsets[2] = vec2( 0.4201, -0.1889);
    offsets[3] = vec2(-0.4063,  0.2191);
    offsets[4] = vec2( 0.1298,  0.4862);
    offsets[5] = vec2(-0.4891, -0.0892);
    offsets[6] = vec2(-0.0943, -0.4918);
    offsets[7] = vec2( 0.5013, -0.0247);

    for (int i = 0; i < PCF_SAMPLES; i++) {
        vec2 off = offsets[i] * radius;
        float s = sin(phi);
        float c = cos(phi);
        vec2 rotOff = vec2(off.x * c - off.y * s, off.x * s + off.y * c);
        sum += texture(shadowTex, vec3(shadowUV + rotOff, shadowNDC.z));
    }
    return sum / float(PCF_SAMPLES);
}

// ── Equirectangular env map lookup ──
vec3 sampleEnvMap(vec3 dir) {
    // dir is world-space reflection direction (Z-up)
    // Convert to equirectangular UV
    // For Z-up: +Z = up (top of env map), -Z = down (bottom)
    // +Y = north (forward in env map), +X = right
    float theta = acos(clamp(dir.z, -1.0, 1.0));
    float phi = atan(dir.y, dir.x); // range [-PI, PI]
    vec2 uv = vec2(phi / (2.0 * PI) + 0.5, theta / PI);
    return texture(envTex, uv).rgb;
}

// ── Evaluate a single directional light ──
vec3 evalLight(vec3 N, vec3 V, vec3 L, vec3 F0, float alpha2, vec3 kD, vec3 baseColor, float intensity, float shadowVis) {
    float NdotV = max(dot(N, V), 0.0001);
    float NdotL = max(dot(N, L), 0.0001);
    vec3 H = normalize(L + V);
    float NdotH = max(dot(N, H), 0.0001);
    float NdotH2 = NdotH * NdotH;

    // GGX distribution
    float denom = NdotH2 * (alpha2 - 1.0) + 1.0;
    float D = alpha2 / (PI * denom * denom);

    // Smith geometry
    float a2_NdotL = alpha2 + (1.0 - alpha2) * NdotL * NdotL;
    float G1_l = 2.0 * NdotL / max(NdotL + sqrt(a2_NdotL), 0.0001);
    float a2_NdotV = alpha2 + (1.0 - alpha2) * NdotV * NdotV;
    float G1_v = 2.0 * NdotV / max(NdotV + sqrt(a2_NdotV), 0.0001);

    // Schlick Fresnel
    vec3 F = F0 + (1.0 - F0) * pow(1.0 - NdotV, 5.0);

    vec3 spec = D * F * G1_l * G1_v / (4.0 * NdotL * NdotV + 0.0001) * specularStrength;
    vec3 diff = baseColor * kD / PI;
    return (diff + spec) * NdotL * intensity * shadowVis;
}

void main() {
    vec3 N = normalize(vWorldNormal);

    vec3 V = normalize(camPos - vWorldPos);
    float NdotV = max(dot(N, V), 0.0001);

    vec3 baseColor = vColor;

    // Fresnel at normal incidence
    vec3 F0 = mix(vec3(0.04), baseColor, metalness);
    float alpha = max(roughness * roughness, 0.001);
    float alpha2 = alpha * alpha;
    vec3 kD = (1.0 - F0) * (1.0 - metalness);

    // ── Direct light (single directional) ──
    vec3 L = normalize(lightDir.xyz);
    float shadowVis = computeShadow(vWorldPos);
    vec3 direct = evalLight(N, V, L, F0, alpha2, kD, baseColor, lightDir.w, shadowVis);

    // ── Ambient (hemisphere + env map) ──
    vec3 ambientUp = vec3(0.6, 0.65, 0.75);
    vec3 ambientDown = vec3(0.1, 0.1, 0.12);
    float NdotUp = dot(N, vec3(0.0, 0.0, 1.0));
    float hemiT = NdotUp * 0.5 + 0.5;
    // Diffuse ambient modulated by surface color
    vec3 ambient = baseColor * mix(ambientDown, ambientUp, hemiT) * 0.4;

    // ── Environment map reflection ──
    vec3 R = reflect(-V, N);
    vec3 envColor = sampleEnvMap(R);
    // Specular IBL: scale by Fresnel at normal incidence
    vec3 envSpec = envColor * (1.0 - roughness) * envIntensity * F0;
    ambient += envSpec * 0.5;

    // Diffuse IBL: rough irradiance from env map
    vec3 envIrradiance = sampleEnvMap(N); // sample in normal direction as cheap irradiance
    vec3 envDiffuse = baseColor * kD * envIrradiance * envIntensity * 0.15;
    ambient += envDiffuse;

    // ── Combine ──
    vec3 lit = ambient * ambientStrength + direct;

    // Simple Reinhard tone mapping
    lit = lit / (lit + 1.0);

    // Gamma correction
    lit = pow(lit, vec3(1.0 / 2.2));

    outColor = vec4(lit, 1.0);
}
