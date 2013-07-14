#include "Common.h"

#if defined(GLSL_VS)
ia_out(GLSL_POSITION)            vec4 ia_VertexPosition;
ia_out(GLSL_NORMAL)              vec4 ia_VertexNormal;
ia_out(GLSL_INSTANCE_TRANSFORM1) mat4 ia_InstanceTransform;
ia_out(GLSL_INSTANCE_PARAM1)     vec4 ia_InstanceParams; // x:radius, y:strength, z:attenuation, w:opacity
#endif
#if defined(GLSL_VS) || defined(GLSL_PS)
vs_out vec4 vs_Params;
vs_out vec4 vs_Outer;
vs_out vec4 vs_Position;
#endif


#if defined(GLSL_VS)

void main()
{
    float strength = ia_InstanceParams.y;
    vec4 instance_pos = ia_InstanceTransform[3];
    vec4 vertex_pos = ia_InstanceTransform * ia_VertexPosition;
    vec3 dir = normalize(vertex_pos.xyz-instance_pos.xyz);
    vec4 outer_pos = vec4(vertex_pos.xyz + dir*strength, 1.0f);

    vs_Params = ia_InstanceParams;
    vs_Outer = u_RS.ModelViewProjectionMatrix * outer_pos;
    vs_Position = u_RS.ModelViewProjectionMatrix * vertex_pos;
    gl_Position = vs_Position;
}

#elif defined(GLSL_PS)

ps_out(0)   vec4 ps_FragColor;

void main()
{
    const int num_samples = 24;
    vec2 center = (1.0+vs_Position.xy / vs_Position.w)*0.5;
    vec2 outer = (1.0+vs_Outer.xy / vs_Outer.w)*0.5;
    vec2 step = (outer - center) / num_samples;

    vec4 color = vec4(0.0);
    for(int i=0; i<num_samples; ++i) {
        color += texture(u_BackBuffer, center - step*i);
    }
    color /= num_samples;
    color.a = vs_Params.w;
    ps_FragColor = color;
}

#endif
