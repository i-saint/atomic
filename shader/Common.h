#pragma include("RenderStates.h")
#pragma include("Semantics.h")


    layout(std140) uniform render_states
    {
        RenderStates u_RS;
    };
#ifdef GLSL_VS
    uniform sampler2D u_ParamBuffer;
#endif // GLSL_VS
#ifdef GLSL_PS
    uniform sampler2D u_ColorBuffer;
    uniform sampler2D u_NormalBuffer;
    uniform sampler2D u_PositionBuffer;
    uniform sampler2D u_GlowBuffer;
    uniform sampler2D u_BackBuffer;
    uniform sampler2D u_RandomBuffer;
    uniform sampler2D u_ParamBuffer;
#endif

#if defined(GLSL_VS)
    #define ia_out(loc) layout(location=loc) in
    #define vs_out out
#elif defined(GLSL)
    #define vs_out in
#endif

#ifdef GLSL_PS
    #define ps_out(loc) layout(location=loc) out
#else
#endif

#if defined(GLSL_GS)
    #define gs_out out
#elif defined(GLSL)
    #define gs_out in
#endif


float SmoothInterpolation2(float v)
{
    return v < 0.5 ? 2.0*v*v : -1.0+4.0*v-2*v*v;
}

float SmoothInterpolation(float v, int n)
{
    return v < 0.5 ? pow(2.0*v, n) : 1.0 - pow(2.0-2.0*v, n);
}
