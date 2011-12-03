#version 410 compatibility

layout(location=0) in vec4 a_VertexPosition;
layout(location=2) in vec2 a_Texcoord;

out vec2 v_Texcoord;

#if defined(GLSL_VS)
#elif defined(GLSL_PS)
#endif


void main()
{
    v_Texcoord = gl_MultiTexCoord0.xy;
    gl_Position = gl_ModelViewProjectionMatrix * a_VertexPosition;
}
