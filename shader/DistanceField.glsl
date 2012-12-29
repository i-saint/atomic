#pragma include("Common.h")

#ifdef GLSL_VS
ia_out(GLSL_POSITION)           vec4 ia_VertexPosition;
ia_out(GLSL_INSTANCE_POSITION)  vec4 ia_InstancePosition;
ia_out(GLSL_INSTANCE_PARAM)     vec4 ia_InstanceParam;
#endif
#if defined(GLSL_VS) || defined(GLSL_PS)
vs_out vec4 vs_VertexColor;
#endif

#if defined(GLSL_VS)

void main()
{
    vec4 vert = ia_VertexPosition+ia_InstancePosition;

    vs_VertexColor = vec4(1.0f-(ia_InstanceParam.w*10.0), 0.0, 0.0, 0.2);
    gl_Position = u_RS.ModelViewProjectionMatrix * vert;
}

#elif defined(GLSL_PS)

ps_out(0) vec4 ps_FlagColor;

void main()
{
    ps_FlagColor    = vs_VertexColor;
}

#endif

