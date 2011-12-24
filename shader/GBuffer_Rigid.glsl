#version 330 core
#pragma include("Common.glslh")

#ifdef GLSL_VS
ia_out(GLSL_POSITION)           vec4 ia_VertexPosition;
ia_out(GLSL_NORMAL)             vec3 ia_VertexNormal;
ia_out(GLSL_INSTANCE_PARAM)     vec4 ia_InstanceParam;
ia_out(GLSL_INSTANCE_POSITION)  vec4 ia_InstancePosition;
ia_out(GLSL_INSTANCE_NORMAL)    vec4 ia_InstanceNormal;
#endif
#if defined(GLSL_VS) || defined(GLSL_PS)
vs_out vec4 vs_VertexPosition;
vs_out vec4 vs_VertexNormal;        // w = shininess
vs_out vec4 vs_VertexColor;         // w = fresnel
vs_out float vs_GlowIntensity;
#endif

#if defined(GLSL_VS)

void main()
{
    vec4 fractionPos = ia_InstancePosition;
    vec4 vert = ia_VertexPosition+fractionPos;
    vert.w = 1.0;

    float dif = length(ia_InstanceNormal.xyz);
    float dif2 = dif*dif;
    float dif3 = dif2*dif;
    float ndif = 1.0-dif;
    vs_GlowIntensity = ndif;
    vs_VertexPosition = vec4(vert.xyz, 1.0);
    vs_VertexNormal = vec4(ia_VertexNormal, 120.0);
    vs_VertexColor = vec4(0.6*dif3, 0.6*dif3, 0.6*dif3, 1.0);
    gl_Position = u_RS.ModelViewProjectionMatrix * vert;
}

#elif defined(GLSL_PS)

ps_out(0) vec4 ps_FlagColor;
ps_out(1) vec4 ps_FragNormal;
ps_out(2) vec4 ps_FragPosition;
ps_out(3) vec4 ps_FragGlow;

void main()
{
    ps_FlagColor    = vs_VertexColor;
    ps_FlagColor.r += vs_GlowIntensity;
    ps_FragNormal   = vs_VertexNormal;
    ps_FragPosition = vs_VertexPosition;
    ps_FragGlow     = vec4(vs_GlowIntensity, 0.0f, 0.0f, 1.0);
}

#endif

