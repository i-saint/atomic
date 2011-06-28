#version 410 compatibility

in vec3 VertexPosition;
in vec3 VertexNormal;
in vec4 VertexColor;

layout(location=0) out vec4 FragColor;
layout(location=1) out vec4 FragNormal;
layout(location=2) out vec4 FragPosition;


void main()
{
    FragColor = VertexColor;
    FragNormal = vec4(VertexNormal, 1.0);
    FragPosition = vec4(VertexPosition, 1.0);
}
