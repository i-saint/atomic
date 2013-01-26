#include "Common.h"

layout(std140) uniform multiresolution_params
{
    MultiresolutionParams u_MR;
};
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

ps_out(0)   vec4 ps_FragColor;

void main()
{
    ivec2 coord = ivec2(gl_FragCoord.x*0.5, gl_FragCoord.y*0.5);
    vec3 Normal0 = texelFetch(u_NormalBuffer, coord+ivec2(0,0), u_MR.Level.x).xyz;
    vec3 Normal1 = texelFetch(u_NormalBuffer, coord+ivec2(1,0), u_MR.Level.x).xyz;
    if(dot(Normal0, Normal1) < u_MR.Threshold.x) {
        discard;
    }

    gl_FragDepth = 0.0;
    ps_FragColor = texture(u_BackBuffer, vs_Texcoord);
}

#endif
