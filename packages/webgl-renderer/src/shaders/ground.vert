#version 300 es
// ground.vert — GLSL ES 3.0
// Infinite ground plane vertex shader.

precision highp float;

layout(location = 0) in vec3 position;

layout(std140) uniform Camera {
    mat4 viewProj;
    mat4 viewMat;
    vec3 camPos;
};

out vec3 vWorldPos;
out vec4 vClipPos;

void main() {
    vec4 worldPos = vec4(position, 1.0);
    gl_Position = viewProj * worldPos;
    vWorldPos = position;
    vClipPos = gl_Position;
}
