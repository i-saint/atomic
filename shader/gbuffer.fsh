#version 410 compatibility

in vec3 v_VertexPosition;
in vec3 v_VertexNormal;
in vec4 v_VertexColor;

layout(location=0) out vec4 o_FragColor;
layout(location=1) out vec4 o_FragNormal;
layout(location=2) out vec4 o_FragPosition;


void main()
{
    o_FragColor = v_VertexColor;
    o_FragNormal = vec4(v_VertexNormal, 1.0);
    o_FragPosition = vec4(v_VertexPosition, 1.0);
}
