#version 330 core
#pragma include("Common.h")

#ifdef GLSL_VS
ia_out(GLSL_POSITION)           vec4 ia_VertexPosition;
ia_out(GLSL_NORMAL)             vec4 ia_VertexNormal;
ia_out(GLSL_TEXCOORD1)          vec2 ia_VertexTexcoord;
#endif
#if defined(GLSL_VS) || defined(GLSL_PS)
vs_out vec4 vs_VertexPosition;      // w = affect bloodstain
vs_out vec4 vs_VertexNormal;        // w = fresnel
vs_out vec4 vs_VertexColor;         // w = shininess
#endif

#if defined(GLSL_VS)

void main()
{
    vs_VertexPosition = ia_VertexPosition;
    vs_VertexNormal = ia_VertexNormal;
    vs_VertexColor = vec4(0.2, 0.2, 0.2, 120.0);
    gl_Position = u_RS.ModelViewProjectionMatrix * vs_VertexPosition;
}

#elif defined(GLSL_PS)

ps_out(0) vec4 ps_FlagColor;
ps_out(1) vec4 ps_FragNormal;
ps_out(2) vec4 ps_FragPosition;

void main()
{
    ps_FlagColor    = vs_VertexColor;
    ps_FragNormal   = vs_VertexNormal;
    ps_FragPosition = vs_VertexPosition;
}

#endif

