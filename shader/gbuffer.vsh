#version 410 compatibility

layout(location=0) in vec4 a_VertexPosition;
layout(location=1) in vec3 a_VertexNormal;
layout(location=2) in vec4 a_InstancePosition;
layout(location=3) in vec4 a_InstanceGlow;

out vec3 v_VertexPosition;
out vec3 v_VertexNormal;
out vec4 v_VertexColor;

void main(void)
{
    vec4 fractionPos = a_InstancePosition;
    vec4 vert = a_VertexPosition+fractionPos;
    vert.w = 1.0;

    v_VertexPosition = vert.xyz;
    v_VertexNormal = a_VertexNormal;
    v_VertexColor = vec4(0.6, 0.6, 0.6, 1.0);

    gl_Position = gl_ModelViewProjectionMatrix * vert;
}
