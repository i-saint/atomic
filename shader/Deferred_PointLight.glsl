#version 330 core
#pragma include("Common.glslh")

#if defined(GLSL_VS)
ia_out(GLSL_POSITION)           vec4 ia_VertexPosition;
ia_out(GLSL_NORMAL)             vec4 ia_VertexNormal;
ia_out(GLSL_INSTANCE_POSITION)  vec4 ia_InstancePosition;
#endif
#if defined(GLSL_VS) || defined(GLSL_PS)
vs_out vec4 vs_LightPosition;
vs_out vec4 vs_LightPositionMVP;
vs_out vec4 vs_LightColor;
vs_out vec4 vs_VertexPositionMVP;
#endif
const float u_LightSize = 1.0;
const float u_LightRange = u_LightSize*0.98;
const float u_LightRange2 = u_LightRange*u_LightRange;
const float u_RcpLightRange2 = 1.0/u_LightRange2;

#if defined(GLSL_VS)

void main()
{
    vs_LightPosition = ia_InstancePosition*1.1;
    vs_LightPosition.z += 0.1;
    vs_LightPosition.w = 0.0;

    vec4 scaled_position = ia_VertexPosition * vec4(u_LightSize, u_LightSize, u_LightSize, 1.0);

    vs_LightColor = vec4(0.1, 0.1, 0.2, 0.1)+normalize(vs_LightPosition)*0.7;
    vs_LightPositionMVP = u_RS.ModelViewProjectionMatrix * vs_LightPosition;
    vs_VertexPositionMVP = u_RS.ModelViewProjectionMatrix * (scaled_position+vs_LightPosition);
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

    vec4 frag_position = texture(u_PositionBuffer, coord);

    vec3 Albedo     = texture(u_ColorBuffer, coord).xyz;
    vec3 Ambient    = normalize(frag_position.xyz)*0.05;
    vec3 Normal     = texture(u_NormalBuffer, coord).xyz;
    vec3 L          = normalize(vs_LightPosition.xyz - frag_position.xyz);
    vec4 color;
    color.xyz = vs_LightColor.xyz * (
                    Ambient +
                    Albedo * max(dot(Normal,L), 0.0)
                 );

    vec3 diff = frag_position.xyz-vs_LightPosition.xyz;
    float dist2 = dot(diff, diff);
    float strength = max(u_LightRange2-dist2, 0.0)*u_RcpLightRange2 * 1.5;
    color.rgb *= pow(strength, 0.7);
    color.w = 1.0;

    ps_FragColor = color;
}

#endif
