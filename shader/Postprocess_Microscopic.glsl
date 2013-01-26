#include "Common.h"

#ifdef GLSL_VS
ia_out(GLSL_POSITION)           vec2 ia_VertexPosition;
ia_out(GLSL_TEXCOORD0)          vec2 ia_VertexTexcoord0;
#endif
#if defined(GLSL_VS) || defined(GLSL_PS)
vs_out vec2 vs_Texcoord;
#endif

#if defined(GLSL_VS)

void main(void)
{
    vs_Texcoord = ia_VertexTexcoord0 * u_RS.ScreenTexcoord;
    gl_Position = vec4(ia_VertexPosition, 0.0, 1.0);
}

#elif defined(GLSL_PS)

ps_out(0) vec4 ps_FragColor;

void main()
{
    vec4 NS         = texture(u_NormalBuffer, vs_Texcoord);
    vec3 Normal     = NS.xyz;
    vec3 FragPos    = texture(u_PositionBuffer, vs_Texcoord).xyz;
    vec3 EyePos     = u_RS.CameraPosition.xyz;
    vec3 EyeDir     = normalize(EyePos - FragPos);

    float d = dot(Normal, EyeDir);
    d = 1.0 - (d *d);
    vec4 color = texture(u_ColorBuffer, vs_Texcoord);
    color.a = dot(color.rgb, vec3(0.299, 0.587, 0.114)) * d;
    ps_FragColor = vec4(color.aaa, 1.0);
}

#endif
