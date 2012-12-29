#pragma include("Common.h")

#if defined(GLSL_VS)
ia_out(GLSL_POSITION)           vec4 ia_VertexPosition;
ia_out(GLSL_TEXCOORD0)          vec2 ia_VertexTexcoord0;
#endif
#if defined(GLSL_VS) || defined(GLSL_PS)
vs_out vec2 vs_Texcoord;
#endif
layout(std140) uniform debug_params
{
    DebugShowBufferParams u_Params;
};

#if defined(GLSL_VS)

void main()
{
    vec4 vertices[4] = vec4[](
        vec4(u_Params.UpperRight.x, u_Params.UpperRight.y, 0.0f, 1.0f),
        vec4(u_Params.BottomLeft.x, u_Params.UpperRight.y, 0.0f, 1.0f),
        vec4(u_Params.BottomLeft.x, u_Params.BottomLeft.y, 0.0f, 1.0f),
        vec4(u_Params.UpperRight.x, u_Params.BottomLeft.y, 0.0f, 1.0f) );
    vec2 texcoords[4] = vec2[](
        vec2(1.0, 1.0),
        vec2(0.0, 1.0),
        vec2(0.0, 0.0),
        vec2(1.0, 0.0) );
    vs_Texcoord = texcoords[gl_VertexID]*u_RS.ScreenTexcoord;
    gl_Position = vertices[gl_VertexID];
}

#elif defined(GLSL_PS)

ps_out(0)   vec4 ps_FragColor;

void main()
{
    vec4 Result = vec4(0.0, 0.0, 0.0, 1.0);
    Result.rgb += clamp(texture(u_ColorBuffer, vs_Texcoord).aaa, u_Params.ColorRange.xxx, u_Params.ColorRange.yyy);
    ps_FragColor = Result;
}

#endif
