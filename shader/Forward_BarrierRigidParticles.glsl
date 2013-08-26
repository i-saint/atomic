#include "Common.h"

#ifdef GLSL_VS
ia_out(GLSL_POSITION)           vec4 ia_VertexPosition;
ia_out(GLSL_NORMAL)             vec3 ia_VertexNormal;
ia_out(GLSL_INSTANCE_POSITION)  vec3 ia_InstancePosition;
ia_out(GLSL_INSTANCE_NORMAL)    vec4 ia_InstanceNormal;
ia_out(GLSL_INSTANCE_PARAM)     int  ia_InstanceID;
#endif
#if defined(GLSL_VS) || defined(GLSL_PS)
vs_out vec4 vs_RefCoord;
vs_out vec4 vs_Normal;
vs_out float vs_Alpha;
#endif
const float scale = 1.5;

#if defined(GLSL_VS)

void main()
{
    vec4 ia_InstanceColor   = texelFetch(u_ParamBuffer, ivec2(0, ia_InstanceID), 0);
    vec4 ia_InstanceGlow    = texelFetch(u_ParamBuffer, ivec2(1, ia_InstanceID), 0);
    vec4 ia_InstanceFlash   = texelFetch(u_ParamBuffer, ivec2(2, ia_InstanceID), 0);
    vec4 vs_InstanceParams = texelFetch(u_ParamBuffer, ivec2(3, ia_InstanceID), 0);

    mat4 trans;
    trans[0] = texelFetch(u_ParamBuffer, ivec2(4, ia_InstanceID), 0);
    trans[1] = texelFetch(u_ParamBuffer, ivec2(5, ia_InstanceID), 0);
    trans[2] = texelFetch(u_ParamBuffer, ivec2(6, ia_InstanceID), 0);
    trans[3] = texelFetch(u_ParamBuffer, ivec2(7, ia_InstanceID), 0);
    vec4 vs_PSetPosition = trans[3];
    
    mat4 rot;
    rot[0] = texelFetch(u_ParamBuffer, ivec2( 8, ia_InstanceID), 0);
    rot[1] = texelFetch(u_ParamBuffer, ivec2( 9, ia_InstanceID), 0);
    rot[2] = texelFetch(u_ParamBuffer, ivec2(10, ia_InstanceID), 0);
    rot[3] = texelFetch(u_ParamBuffer, ivec2(11, ia_InstanceID), 0);

    vec4 params = texelFetch(u_ParamBuffer, ivec2(3, ia_InstanceID), 0);
    vs_Alpha = min(params.x*0.005, 1.0);

    vec4 instancePos = trans * vec4(ia_InstancePosition.xyz, 1.0);
    vec4 vertexPos = rot * vec4(ia_VertexPosition.xyz*scale*vs_InstanceParams.z, 0.0);
    vec4 vert = vec4(vertexPos.xyz+instancePos.xyz, 1.0);
    vec3 n = normalize((rot * vec4(ia_VertexNormal.xyz, 0.0)).xyz);
    vec4 rp = u_RS.ModelViewProjectionMatrix * (vert + vec4(n.xyz*0.2*vs_Alpha, 0.0));

    vs_RefCoord = rp;
    vs_Normal= vec4(n,0.0);
    gl_Position = u_RS.ModelViewProjectionMatrix * vert;
}

#elif defined(GLSL_PS)

ps_out(0) vec4 ps_FlagColor;

void main()
{
    vec2 coord = (1.0+vs_RefCoord.xy / vs_RefCoord.w)*0.5;
    vec4 color = texture(u_BackBuffer, coord + vec2( 0.0, 0.0)*u_RS.RcpScreenSize);
    float s = 1.0-abs(dot(vs_Normal.xyz, u_RS.CameraDirection.xyz));
    float scanline = mod(gl_FragCoord.y+u_RS.Frame*0.7, 2.0)*0.02;
    ps_FlagColor = vec4(color.xyz+vec3(s*0.2+scanline)*vs_Alpha, s);
}

#endif

