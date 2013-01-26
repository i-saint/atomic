#include "Common.h"

#if defined(GLSL_VS)
ia_out(GLSL_POSITION)           vec4 ia_VertexPosition;
ia_out(GLSL_NORMAL)             vec4 ia_VertexNormal;
ia_out(GLSL_INSTANCE_POSITION)  vec4 ia_InstancePosition;
ia_out(GLSL_INSTANCE_COLOR)     vec4 ia_InstanceColor;
ia_out(GLSL_INSTANCE_PARAM)     vec4 ia_InstanceParam;
#endif
#if defined(GLSL_VS) || defined(GLSL_PS)
vs_out vec4 vs_LightPosition;
vs_out vec4 vs_LightPositionMVP;
vs_out vec4 vs_LightColor;
vs_out vec4 vs_VertexPositionMVP;
vs_out float vs_LightRange;
vs_out float vs_RcpLightRange;
#endif

#if defined(GLSL_VS)

void main()
{
    vs_LightPosition = ia_InstancePosition;
    vs_LightPosition.w = 1.0;
    vs_LightRange = ia_InstanceParam.x;
    vs_RcpLightRange = ia_InstanceParam.y;

    vec4 scaled_position = ia_VertexPosition * vec4(vs_LightRange, vs_LightRange, vs_LightRange, 0.0);

    vs_LightColor = ia_InstanceColor;
    vs_LightPositionMVP = u_RS.ModelViewProjectionMatrix * vs_LightPosition;
    vs_VertexPositionMVP= u_RS.ModelViewProjectionMatrix * (scaled_position+vs_LightPosition);
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

    vec3 FragPos    = texture(u_PositionBuffer, coord).xyz;
    vec3  LightColor    = vs_LightColor.rgb;
    vec3  LightPos      = vs_LightPosition.xyz;
    vec3  LightDiff     = vs_LightPosition.xyz - FragPos.xyz;
    float LightDist2    = dot(LightDiff, LightDiff);
    float LightDist     = sqrt(LightDist2);
    vec3  LightDir      = LightDiff / LightDist;
    float LightAttenuation  = max(vs_LightRange-LightDist, 0.0)*vs_RcpLightRange * 1.5;
    if(LightAttenuation==0.0) { discard; }

    {
        vec2 lcoord;
        lcoord.x = (1.0 + (vs_LightPositionMVP.x/vs_LightPositionMVP.w))*0.5;
        lcoord.y = (1.0 + (vs_LightPositionMVP.y/vs_LightPositionMVP.w))*0.5;
        lcoord *= u_RS.ScreenTexcoord;

        const int Div = 50;
        vec2 D2 = (coord - lcoord) / Div;
        vec3 D3 = (FragPos - vs_LightPosition.xyz) / Div;
        float LightAttenuationMax = LightAttenuation;
        for(int i=0; i<Div; ++i) {
            vec4 RayPos = vs_LightPosition + vec4(D3*i, 0.0);
            vec3 RayFragPos = texture(u_PositionBuffer, lcoord + (D2*i)).xyz;
            if(RayPos.z < RayFragPos.z) {
                LightAttenuation -= 0.03;
            }
            else {
                LightAttenuation = min(LightAttenuation+0.003, LightAttenuationMax);
            }
        }
        if(LightAttenuation<=0.0) { discard; }
    }

    vec4 AS         = texture(u_ColorBuffer, coord);
    vec4 NS         = texture(u_NormalBuffer, coord);
    vec3 Albedo     = AS.rgb;
    float Shininess = AS.a;
    float Fresnel   = NS.a;
    vec3 Normal     = NS.xyz;
    vec3 EyePos     = u_RS.CameraPosition.xyz;
    vec3 EyeDir     = normalize(EyePos - FragPos);

    vec3 h          = normalize(EyeDir + LightDir);
    float nh        = max(dot(Normal, h), 0.0);
    float Specular  = pow(nh, Shininess);
    float Intensity = max(dot(Normal, LightDir), 0.0);

    vec3 Ambient    = normalize(FragPos)*0.05;
    vec4 Result = vec4(0.0, 0.0, 0.0, 1.0);
    Result.rgb += vs_LightColor.rgb * (Ambient + Albedo * Intensity) * LightAttenuation;
    Result.rgb += vs_LightColor.rgb * Specular * LightAttenuation;

    ps_FragColor = Result;
}

#endif
