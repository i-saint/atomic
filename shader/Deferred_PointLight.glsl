#version 330 compatibility
#pragma include("Common.glslh")

#if defined(GLSL_VS)
ia_out(GLSL_POSITION)           vec4 ia_VertexPosition;
ia_out(GLSL_INSTANCE_POSITION)  vec4 ia_InstancePosition;
#endif
#if defined(GLSL_VS) || defined(GLSL_PS)
vs_out vec4 vs_LightPosition;
vs_out vec4 vs_LightPositionMVP;
vs_out vec4 vs_LightColor;
vs_out vec4 vs_VertexPositionMVP;
#endif

#if defined(GLSL_VS)

void main()
{
    vs_LightPosition = ia_InstancePosition*1.1;
    vs_LightPosition.z += 0.1;
    vs_LightPosition.w = 0.0;

    vs_LightColor = vec4(0.1, 0.1, 0.2, 0.1)+normalize(vs_LightPosition)*0.7;
    vs_LightPositionMVP = gl_ModelViewProjectionMatrix * vs_LightPosition;
    vs_VertexPositionMVP = gl_ModelViewProjectionMatrix * (ia_VertexPosition+vs_LightPosition);
    gl_Position = vs_VertexPositionMVP;
}

#elif defined(GLSL_PS)

uniform sampler2D u_ColorBuffer;
uniform sampler2D u_NormalBuffer;
uniform sampler2D u_PositionBuffer;
uniform float u_RcpAspectRatio;
uniform vec2 u_TexcoordScale;

ps_out(0)   vec4 ps_FragColor;

void main()
{
    vec2 coord;
    coord.x = (1.0 + (vs_VertexPositionMVP.x/vs_VertexPositionMVP.w))*0.5;
    coord.y = (1.0 + (vs_VertexPositionMVP.y/vs_VertexPositionMVP.w))*0.5* u_RcpAspectRatio;
    coord *= u_TexcoordScale;

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

    float strength = max(2.3-length(frag_position.xyz-vs_LightPosition.xyz), 0.0)/2.3 * 1.5;
    color.rgb *= pow(strength, 0.7);
    color.w = 1.0;

    ps_FragColor = color;

    ps_FragColor.w = 1.0;
}

#endif
