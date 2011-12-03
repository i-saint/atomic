#define GLSL_POSITION			0
#define GLSL_NORMAL				1
#define GLSL_TEXCOORD0			2

#define GLSL_INSTANCE_POSITION	8
#define GLSL_INSTANCE_NORMAL	9
#define GLSL_INSTANCE_GLOW		10
#define GLSL_STATES             0

struct GLSLStates
{
    mat4 ModelViewProjectionMatrix;
    vec4 CameraPosition;
    float ScreenWidth;
    float ScreenHeight;
    float RcpScreenWidth;
    float RcpScreenHeight;
};


#ifdef GLSL
    uniform globals
    {
        GLSLStates u_Globals;
    };
#endif

#if defined(GLSL_VS)
    #define ia_out(loc) layout(location=loc) in
    #define vs_out out
#elif defined(GLSL)
    #define vs_out in
#endif

#if defined(GLSL_GS)
    #define gs_out out
#elif defined(GLSL)
    #define gs_out in
#endif

#ifdef GLSL_PS
    #define ps_out(loc) layout(location=loc) out
#else
#endif
