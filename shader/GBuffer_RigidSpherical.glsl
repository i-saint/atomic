#pragma include("Common.h")

#ifdef GLSL_VS
ia_out(GLSL_POSITION)           vec4 ia_VertexPosition;
ia_out(GLSL_NORMAL)             vec3 ia_VertexNormal;
ia_out(GLSL_INSTANCE_POSITION)  vec3 ia_InstancePosition;
ia_out(GLSL_INSTANCE_NORMAL)    vec4 ia_InstanceNormal;
ia_out(GLSL_INSTANCE_PARAM)     int  ia_InstanceID;
#endif
#if defined(GLSL_VS) || defined(GLSL_PS)
vs_out vec4 vs_InstancePosition;
vs_out vec4 vs_VertexPosition;      // w = affect bloodstain
vs_out vec4 vs_VertexNormal;        // w = fresnel
vs_out vec4 vs_VertexColor;         // w = shininess
vs_out vec4 vs_Glow;
vs_out vec4 vs_Flash;
#endif

#if defined(GLSL_VS)

void main()
{
    vec4 ia_InstanceColor   = texelFetch(u_ParamBuffer, ivec2(0, ia_InstanceID), 0);
    vec4 ia_InstanceGlow    = texelFetch(u_ParamBuffer, ivec2(1, ia_InstanceID), 0);
    vec4 ia_InstanceFlash   = texelFetch(u_ParamBuffer, ivec2(2, ia_InstanceID), 0);

    mat4 trans;
    trans[0] = texelFetch(u_ParamBuffer, ivec2(4, ia_InstanceID), 0);
    trans[1] = texelFetch(u_ParamBuffer, ivec2(5, ia_InstanceID), 0);
    trans[2] = texelFetch(u_ParamBuffer, ivec2(6, ia_InstanceID), 0);
    trans[3] = texelFetch(u_ParamBuffer, ivec2(7, ia_InstanceID), 0);

    vec4 instancePos = trans * vec4(ia_InstancePosition, 1.0);
    vec4 vert = ia_VertexPosition+instancePos;
    vert.w = 1.0;

    float dif = ia_InstanceNormal.w;
    float dif2 = dif*dif;
    float dif3 = dif2*dif;
    float ndif = 1.0-dif;
    vs_Glow = ia_InstanceGlow * (ndif*ndif);
    vs_Flash = ia_InstanceFlash;

    vs_VertexPosition = vec4(vert.xyz, 1.0);
    vs_VertexNormal = vec4(ia_VertexNormal, 0.04);
    vs_VertexColor = vec4(ia_InstanceColor.rgb*dif3, ia_InstanceColor.a);
    vs_InstancePosition = instancePos;
    gl_Position = u_RS.ModelViewProjectionMatrix * vert;
}

#elif defined(GLSL_PS)

ps_out(0) vec4 ps_FlagColor;
ps_out(1) vec4 ps_FragNormal;
ps_out(2) vec4 ps_FragPosition;
ps_out(3) vec4 ps_FragGlow;

void main()
{
    const float radius = 0.015f;
    vec2 pos2 = vs_VertexPosition.xy - vs_InstancePosition.xy;
    if(dot(pos2, pos2) > radius*radius) {
        discard;
    }
    float z = sqrt(radius*radius - pos2.x*pos2.x - pos2.y*pos2.y);

    vec3 n = normalize(vs_VertexPosition.xyz - vs_InstancePosition.xyz);
    ps_FlagColor    = vs_VertexColor + vs_Glow;
    ps_FragNormal   = vec4(n, vs_VertexNormal.w);
    ps_FragPosition = vs_InstancePosition + vec4(pos2, z, 0.0);
    ps_FragGlow     = vec4(vs_Glow.rgb + vs_Flash.rgb, 1.0);
}

#endif

