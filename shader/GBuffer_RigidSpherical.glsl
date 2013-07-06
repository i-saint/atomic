#include "Common.h"

#ifdef GLSL_VS
ia_out(GLSL_POSITION)           vec4 ia_VertexPosition;
ia_out(GLSL_NORMAL)             vec3 ia_VertexNormal;
ia_out(GLSL_INSTANCE_POSITION)  vec3 ia_InstancePosition;
ia_out(GLSL_INSTANCE_NORMAL)    vec4 ia_InstanceNormal;
ia_out(GLSL_INSTANCE_PARAM)     int  ia_InstanceID;
#endif
#if defined(GLSL_VS) || defined(GLSL_PS)
vs_out vec4 vs_PSetPosition;
vs_out vec4 vs_InstancePosition;
vs_out vec4 vs_InstanceParams;      // x: elapsed frame, y: appear_radius
vs_out vec4 vs_VertexPosition;      // w = affect bloodstain
vs_out vec4 vs_VertexNormal;        // w = fresnel
vs_out vec4 vs_VertexColor;         // w = shininess
vs_out vec4 vs_Glow;
vs_out vec4 vs_Flash;
#endif
const float scale = 1.2;

#if defined(GLSL_VS)

void main()
{
    vec4 ia_InstanceColor   = texelFetch(u_ParamBuffer, ivec2(0, ia_InstanceID), 0);
    vec4 ia_InstanceGlow    = texelFetch(u_ParamBuffer, ivec2(1, ia_InstanceID), 0);
    vec4 ia_InstanceFlash   = texelFetch(u_ParamBuffer, ivec2(2, ia_InstanceID), 0);
    vs_InstanceParams = texelFetch(u_ParamBuffer, ivec2(3, ia_InstanceID), 0);

    mat4 trans;
    trans[0] = texelFetch(u_ParamBuffer, ivec2(4, ia_InstanceID), 0);
    trans[1] = texelFetch(u_ParamBuffer, ivec2(5, ia_InstanceID), 0);
    trans[2] = texelFetch(u_ParamBuffer, ivec2(6, ia_InstanceID), 0);
    trans[3] = texelFetch(u_ParamBuffer, ivec2(7, ia_InstanceID), 0);
    vs_PSetPosition = trans[3];

    vec4 instancePos = trans * vec4(ia_InstancePosition, 1.0);
    vec4 vert = ia_VertexPosition*vs_InstanceParams.z*scale + instancePos;
    vert.w = 1.0;

    float dif = ia_InstanceNormal.w;
    float dif2 = dif*dif;
    float dif3 = dif2*dif;
    float ndif = 1.0-dif;
    vs_Glow = ia_InstanceGlow * (ndif*ndif);
    vs_Flash = ia_InstanceFlash;

    vs_VertexPosition = vec4(vert.xyz, 1.0);
    vs_VertexNormal = vec4(ia_VertexNormal, 0.04);
    vs_VertexColor = vec4(ia_InstanceColor.rgb*dif3, ia_InstanceColor.a);
    vs_InstancePosition = instancePos;
    gl_Position = u_RS.ModelViewProjectionMatrix * vert;
}

#elif defined(GLSL_PS)

ps_out(0) vec4 ps_FlagColor;
ps_out(1) vec4 ps_FragNormal;
ps_out(2) vec4 ps_FragPosition;
ps_out(3) vec4 ps_FragGlow;

void main()
{
    float radius = 0.015f*vs_InstanceParams.z * scale;
    vec2 diff2 = vs_VertexPosition.xy - vs_InstancePosition.xy;
    if(dot(diff2, diff2) > radius*radius) {
        discard;
    }

    float z = sqrt(radius*radius - diff2.x*diff2.x - diff2.y*diff2.y);
    vec4 flag_pos = vs_InstancePosition + vec4(diff2, z, 0.0);
    vec4 glow = vec4(vs_Glow.rgb + vs_Flash.rgb, 1.0);

    {
        // 出現エフェクト
        float ar = vs_InstanceParams.y;
        vec3 psetpos = vs_PSetPosition.xyz;
        vec3 diff3 = flag_pos.xyz - psetpos;
        float d = length(diff3);
        if(d > ar) {
            discard;
        }
        float cr = max(0.0, 1.0f - (ar-d)*40.0);
        float co = max(0.0, 1.0f - (ar-d)*60.0);
        glow += vec4(1.0, 0.6, 0.7, 0.0) * vec4(cr, co, co, 0.0);
    }

    vec3 n = normalize(vs_VertexPosition.xyz - vs_InstancePosition.xyz);
    ps_FlagColor    = vs_VertexColor + vs_Glow;
    ps_FragNormal   = vec4(n, vs_VertexNormal.w);
    ps_FragPosition = flag_pos;
    ps_FragGlow     = glow;
}

#endif

