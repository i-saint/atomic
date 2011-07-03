#version 410 compatibility

layout(location=0) in vec4 a_VertexPosition;
layout(location=2) in vec4 a_InstancePosition;

out vec4 v_LightPosition;
out vec4 v_LightPositionMVP;
out vec4 v_LightColor;
out vec4 v_VertexPositionMVP;


void main()
{
    v_LightPosition = a_InstancePosition*1.1;

    v_LightColor = normalize(v_LightPosition);
    v_LightPositionMVP = gl_ModelViewProjectionMatrix * v_LightPosition;
    v_VertexPositionMVP = gl_ModelViewProjectionMatrix * (a_VertexPosition+v_LightPosition);
    gl_Position = v_VertexPositionMVP;
}
