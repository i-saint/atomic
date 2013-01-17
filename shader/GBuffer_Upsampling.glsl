#pragma include("Common.h")

#if defined(GLSL_VS)
ia_out(GLSL_POSITION)           vec4 ia_VertexPosition;
ia_out(GLSL_TEXCOORD0)          vec2 ia_VertexTexcoord0;
#endif
#if defined(GLSL_VS) || defined(GLSL_PS)
vs_out vec2 vs_Texcoord;
#endif

#if defined(GLSL_VS)

void main()
{
    vs_Texcoord = ia_VertexTexcoord0*u_RS.ScreenTexcoord;
    gl_Position = ia_VertexPosition;
}

#elif defined(GLSL_PS)

ps_out(0) vec4 ps_FlagColor;
ps_out(1) vec4 ps_FragNormal;
ps_out(2) vec4 ps_FragPosition;
ps_out(3) vec4 ps_FragGlow;

void main()
{
    const int level = 2;
    const float diff = 0.01;
    ivec2 coord = ivec2(gl_FragCoord.x*0.25, gl_FragCoord.y*0.25);
    vec4 pos1 = texelFetch(u_PositionBuffer, coord+ivec2( 0,0), level);
    vec4 pos2 = texelFetch(u_PositionBuffer, coord+ivec2( 1,0), level);
    vec4 pos3 = texelFetch(u_PositionBuffer, coord+ivec2( 0,1), level);
    if(abs(pos2.w-pos1.w) > diff || abs(pos3.w-pos1.w) > diff) {
        discard;
    }

    gl_FragDepth    = 0.0;
    ps_FlagColor    = texelFetch(u_ColorBuffer, coord, level);
    ps_FragNormal   = texelFetch(u_NormalBuffer, coord, level);
    ps_FragPosition = texelFetch(u_PositionBuffer, coord, level);

    vec4 g = texelFetch(u_GlowBuffer, coord, level);
    if(u_RS.ShowMultiresolution!=0) {
        g = vec4(0.5, 0.0, 0.0, 0.0);
    }
    ps_FragGlow     = g;
}

#endif
