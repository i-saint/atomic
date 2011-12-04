#version 330 core
#pragma include("Common.glslh")

#if defined(GLSL_VS) || defined(GLSL_PS)
vs_out vec2 vs_Texcoord;
#endif

#if defined(GLSL_VS)


void main(void)
{
    vec2 pos[4] = vec2[4](vec2(1.0, 1.0), vec2(-1.0, 1.0), vec2(-1.0, -1.0), vec2(1.0,-1.0));
    vec2 tex[4] = vec2[4](vec2(1.0, 1.0), vec2(0.0, 1.0), vec2(0.0, 0.0), vec2(1.0, 0.0));
    int indices[6] = int[6](0,1,2, 2,3,0);

    int i = indices[gl_VertexID];
    vs_Texcoord = tex[i] * u_RS.ScreenTexcoord;
    gl_Position = vec4(pos[i], 0.0, 1.0);
}

#elif defined(GLSL_PS)

ps_out(0) vec4 ps_FragColor;

void main()
{
    ps_FragColor = texture(u_RS.ColorBuffer, vs_Texcoord);
}

#endif


