#version 330 core
#pragma include("Common.h")

#ifdef GLSL_VS
ia_out(GLSL_POSITION)           vec4 ia_VertexPosition;
ia_out(GLSL_COLOR)              vec4 ia_VertexColor;
#endif
#if defined(GLSL_VS) || defined(GLSL_PS)
vs_out vec4 vs_VertexColor;
#endif

#if defined(GLSL_VS)

void main()
{
    vs_VertexColor      = ia_VertexColor;
    gl_Position         = u_RS.ModelViewProjectionMatrix * ia_VertexPosition;
}

#elif defined(GLSL_PS)

ps_out(0) vec4 ps_FlagColor;

void main()
{
    ps_FlagColor    = vs_VertexColor;
}

#endif

