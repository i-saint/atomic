#version 330 core
#pragma include("Common.glslh")

#ifdef GLSL_VS
ia_out(GLSL_POSITION)           vec4 ia_VertexPosition;
ia_out(GLSL_NORMAL)             vec3 ia_VertexNormal;
ia_out(GLSL_INSTANCE_POSITION)  vec4 ia_InstancePosition;
ia_out(GLSL_INSTANCE_NORMAL)    vec4 ia_InstanceNormal;
ia_out(GLSL_INSTANCE_COLOR)     vec4 ia_InstanceColor;
ia_out(GLSL_INSTANCE_GLOW)      vec4 ia_InstanceGlow;
ia_out(GLSL_INSTANCE_PARAM)     vec4 ia_InstanceFlash;
#endif
#if defined(GLSL_VS) || defined(GLSL_PS)
vs_out vec4 vs_InstancePosition;
vs_out vec4 vs_VertexPosition;
vs_out vec4 vs_VertexNormal;        // w = shininess
vs_out vec4 vs_VertexColor;         // w = fresnel
vs_out vec4 vs_Glow;
vs_out vec4 vs_Flash;
#endif

#if defined(GLSL_VS)

void main()
{
    vec4 fractionPos = ia_InstancePosition;
    vec4 vert = ia_VertexPosition+fractionPos;
    vert.w = 1.0;

    float dif = ia_InstanceNormal.w;
    float dif2 = dif*dif;
    float dif3 = dif2*dif;
    float ndif = 1.0-dif;
    vs_Glow = ia_InstanceGlow * (ndif*ndif);
    vs_Flash = ia_InstanceFlash;
    vs_VertexPosition = vec4(vert.xyz, 1.0);
    vs_VertexNormal = vec4(ia_VertexNormal, 120.0);
    vs_VertexColor = vec4(ia_InstanceColor.rgb*dif3, 1.0);
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
    vec3 n = normalize(vs_VertexPosition.xyz - vs_InstancePosition.xyz);
    ps_FlagColor    = vs_VertexColor + vs_Glow;
    ps_FragNormal   = vec4(n, vs_VertexNormal.w);
    ps_FragPosition = vs_VertexPosition;
    ps_FragGlow     = vec4(vs_Glow.rgb + vs_Flash.rgb, 1.0);
}

#endif

