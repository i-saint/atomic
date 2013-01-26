#include "Common.h"

#ifdef GLSL_VS
ia_out(GLSL_POSITION)           vec4 ia_VertexPosition;
ia_out(GLSL_NORMAL)             vec3 ia_VertexNormal;
ia_out(GLSL_INSTANCE_POSITION)  vec4 ia_InstancePosition;
ia_out(GLSL_INSTANCE_COLOR)     vec4 ia_InstanceColor;
ia_out(GLSL_INSTANCE_GLOW)      vec4 ia_InstanceGlow;
ia_out(GLSL_INSTANCE_PARAM)     vec4 ia_InstanceParam; // x = scale
#endif
#if defined(GLSL_VS) || defined(GLSL_PS)
vs_out vec4 vs_InstancePosition;
vs_out vec4 vs_VertexPosition;      // w = affect bloodstain
vs_out vec4 vs_VertexNormal;        // w = fresnel
vs_out vec4 vs_VertexColor;         // w = shininess
vs_out vec4 vs_Glow;
vs_out float vs_Scale;
#endif

#if defined(GLSL_VS)

void main()
{
    float scale = ia_InstanceParam.x;
    vec4 vert = (ia_VertexPosition*scale)+ia_InstancePosition;
    vert.w = 1.0;

    vs_Glow = ia_InstanceGlow;
    vs_Scale = scale;

    vs_VertexPosition = vec4(vert.xyz, 1.0);
    vs_VertexNormal = vec4(ia_VertexNormal, 0.04);
    vs_VertexColor = ia_InstanceColor;
    vs_InstancePosition = ia_InstancePosition;
    gl_Position = u_RS.ModelViewProjectionMatrix * vert;
}

#elif defined(GLSL_PS)

ps_out(0) vec4 ps_FlagColor;
ps_out(1) vec4 ps_FragNormal;
ps_out(2) vec4 ps_FragPosition;
ps_out(3) vec4 ps_FragGlow;

void main()
{
    float radius = 0.015 * vs_Scale;
    vec2 pos2 = vs_VertexPosition.xy - vs_InstancePosition.xy;
    if(dot(pos2, pos2) > radius*radius) {
        discard;
    }
    float z = sqrt(radius*radius - pos2.x*pos2.x - pos2.y*pos2.y);

    vec3 n = normalize(vs_VertexPosition.xyz - vs_InstancePosition.xyz);
    ps_FlagColor    = vs_VertexColor + vs_Glow;
    ps_FragNormal   = vec4(n, vs_VertexNormal.w);
    ps_FragPosition = vs_InstancePosition + vec4(pos2, z, 0.0);
    ps_FragGlow     = vec4(vs_Glow.rgb, 1.0);
}

#endif

