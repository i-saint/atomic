#include "Common.h"

#if defined(GLSL_VS)
ia_out(GLSL_POSITION)            vec4 ia_VertexPosition;
ia_out(GLSL_NORMAL)              vec4 ia_VertexNormal;
ia_out(GLSL_INSTANCE_TRANSFORM1) mat4 ia_InstanceTransform;
#endif
#if defined(GLSL_VS) || defined(GLSL_PS)
vs_out vec4 vs_RefCoord;
vs_out vec3 vs_Normal;
#endif

#if defined(GLSL_VS)

void main()
{
    vec4 n = normalize(u_RS.ModelViewProjectionMatrix * (ia_InstanceTransform * vec4(ia_VertexNormal.xyz, 0.0)));
    vec4 rp = u_RS.ModelViewProjectionMatrix * (ia_InstanceTransform * (ia_VertexPosition + vec4(n.xyz*0.3, 0.0)));

    vec4 pos = ia_InstanceTransform * ia_VertexPosition;
    vs_RefCoord = rp;
    vs_Normal = n.xyz;
    gl_Position = u_RS.ModelViewProjectionMatrix * pos;
}

#elif defined(GLSL_PS)

ps_out(0)   vec4 ps_FragColor;

void main()
{
    vec2 coord = (1.0+vs_RefCoord.xy / vs_RefCoord.w)*0.5;
    vec4 color = texture(u_BackBuffer, coord + vec2( 0.0, 0.0)*u_RS.RcpScreenSize);
    float s = 1.0-abs(dot(vs_Normal.xyz, u_RS.CameraDirection.xyz));
    s = max(s-0.4, 0.0)*1.666666;
    ps_FragColor = vec4(color.xyz+s*0.5, s);
}

#endif
