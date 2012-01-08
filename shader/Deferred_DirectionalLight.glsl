#version 330 core
#pragma include("Common.glslh")

#if defined(GLSL_VS)
ia_out(GLSL_POSITION)           vec4 ia_VertexPosition;
ia_out(GLSL_TEXCOORD0)          vec2 ia_VertexTexcoord0;
ia_out(GLSL_INSTANCE_DIRECTION) vec4 ia_InstanceDirection;
ia_out(GLSL_INSTANCE_COLOR)     vec4 ia_InstanceColor;
ia_out(GLSL_INSTANCE_AMBIENT)   vec4 ia_InstanceAmbient;
#endif
#if defined(GLSL_VS) || defined(GLSL_PS)
vs_out vec2 vs_Texcoord;
vs_out vec4 vs_LightDirection;
vs_out vec4 vs_LightColor;
vs_out vec4 vs_LightAmbient;
vs_out vec4 vs_VertexPositionMVP;
#endif

#if defined(GLSL_VS)

void main()
{
    vs_Texcoord = ia_VertexTexcoord0*u_RS.ScreenTexcoord;

    vs_LightDirection   = ia_InstanceDirection;
    vs_LightColor       = ia_InstanceColor;
    vs_LightAmbient     = ia_InstanceAmbient;
    gl_Position         = ia_VertexPosition;
}

#elif defined(GLSL_PS)

ps_out(0)   vec4 ps_FragColor;

void main()
{
    vec2 coord = vs_Texcoord;

    vec4 AS         = texture(u_ColorBuffer, coord);
    vec4 NS         = texture(u_NormalBuffer, coord);
    vec3 Albedo     = AS.rgb;
    float Shininess = AS.a;
    float Fresnel   = NS.a;
    vec3 Normal     = NS.xyz;
    vec3 FragPos    = texture(u_PositionBuffer, coord).xyz;
    vec3 EyePos     = u_RS.CameraPosition.xyz;
    vec3 EyeDir     = normalize(EyePos - FragPos);
    
    vec3 LightColor = vs_LightColor.rgb;
    vec3 LightDir   = -vs_LightDirection.xyz;

    vec3 h          = normalize(EyeDir + LightDir);
    float nh        = max(dot(Normal, h), 0.0);
    float Specular  = pow(nh, Shininess);
    float Intensity = max(dot(Normal, LightDir), 0.0) * 1.5;

    vec3 Ambient    = vs_LightAmbient.rgb;
    vec4 Result = vec4(0.0, 0.0, 0.0, 1.0);
    Result.rgb += vs_LightColor.rgb * (Ambient + Albedo * Intensity);
    Result.rgb += vs_LightColor.rgb * Specular;
    Result.rgb += texture(u_GlowBuffer, coord).rgb;

    ps_FragColor = Result;
}

#endif
