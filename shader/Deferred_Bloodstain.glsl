#include "Common.h"

#if defined(GLSL_VS)
ia_out(GLSL_POSITION)           vec4 ia_VertexPosition;
ia_out(GLSL_INSTANCE_POSITION)  vec4 ia_InstancePosition;
ia_out(GLSL_INSTANCE_PARAM)     vec4 ia_InstanceParam;
#endif
#if defined(GLSL_VS) || defined(GLSL_PS)
vs_out vec4 vs_LightPosition;
vs_out vec4 vs_LightColor;
vs_out vec4 vs_VertexPositionMVP;
vs_out float vs_LightRange;
vs_out float vs_RcpLightRange;
#endif

#if defined(GLSL_VS)

void main()
{
    vs_LightPosition    = ia_InstancePosition;
    vs_LightPosition.w  = 0.0;
    vs_LightRange       = 0.05;
    vs_RcpLightRange    = 20.0;

    vec4 scaled_v = ia_VertexPosition;
    scaled_v.xyz *= vec3(ia_InstanceParam.xxx);
    vs_LightColor = vec4(0.4, 0.0, 0.05, 1.0);
    vs_VertexPositionMVP = u_RS.ModelViewProjectionMatrix * (scaled_v+vs_LightPosition);
    gl_Position = vs_VertexPositionMVP;
}

#elif defined(GLSL_PS)

ps_out(0)   vec4 ps_FragColor;

void main()
{
    vec2 coord;
    coord.x = (1.0 + (vs_VertexPositionMVP.x/vs_VertexPositionMVP.w))*0.5;
    coord.y = (1.0 + (vs_VertexPositionMVP.y/vs_VertexPositionMVP.w))*0.5;
    coord *= u_RS.ScreenTexcoord;

    vec4 PosA       = texture(u_PositionBuffer, coord);
    vec4 NS         = texture(u_NormalBuffer, coord);
    vec3 Normal     = NS.xyz;
    vec3 FragPos    = PosA.xyz;

    vec3  LightColor    = vs_LightColor.rgb;
    vec3  LightPos      = vs_LightPosition.xyz;
    vec3  LightDiff     = vs_LightPosition.xyz - FragPos.xyz;
    float LightDist2    = dot(LightDiff, LightDiff);
    float LightDist     = sqrt(LightDist2);

    float LightAttenuation  = max(vs_LightRange-LightDist, 0.0)*vs_RcpLightRange;
    if(LightAttenuation * PosA.a < 0.01) {
        discard;
    }

    vec4 Result = vec4(0.0, 0.0, 0.0, 120.0);
    Result.rgb += vs_LightColor.rgb;

    ps_FragColor = Result;
}

#endif
