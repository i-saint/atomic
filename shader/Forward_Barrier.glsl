#include "Common.h"

#if defined(GLSL_VS)
ia_out(GLSL_POSITION)            vec4 ia_VertexPosition;
ia_out(GLSL_NORMAL)              vec4 ia_VertexNormal;
ia_out(GLSL_INSTANCE_TRANSFORM1) mat4 ia_InstanceTransform;
#endif
#if defined(GLSL_VS) || defined(GLSL_PS)
vs_out vec3 vs_VertexNormal;
vs_out vec3 vs_VertexPosition;
#endif

#if defined(GLSL_VS)

void main()
{
    vec4 p = u_RS.ModelViewProjectionMatrix * (ia_InstanceTransform * ia_VertexPosition);
    vec4 n = normalize(u_RS.ModelViewProjectionMatrix * (ia_InstanceTransform * vec4(ia_VertexNormal.xyz, 0.0)));
    vs_VertexNormal = n.xyz;
    vs_VertexPosition = p.xyz;
    gl_Position = p;
}

#elif defined(GLSL_PS)

ps_out(0)   vec4 ps_FragColor;

void main()
{
    vec2 pos = gl_FragCoord.xy*u_RS.RcpScreenSize;
    vec4 color = texture(u_BackBuffer, pos-vs_VertexNormal.xy*0.01);
    float s = 1.0 - abs(dot(vs_VertexNormal.xyz, u_RS.CameraDirection.xyz));
    s *= s;
    ps_FragColor = vec4(color.xyz+s*0.1, s);
}

#endif
