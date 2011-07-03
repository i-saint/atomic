#version 410 compatibility

layout(location=0) in vec4 a_VertexPosition;
layout(location=2) in vec2 a_Texcoord;

out vec2 v_Texcoord;


void main()
{
    v_Texcoord = a_Texcoord;
    gl_Position = gl_ModelViewProjectionMatrix * a_VertexPosition;
}
