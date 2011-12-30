#version 330 core
#pragma include("Common.glslh")

layout(std140) uniform fade_params
{
    FadeParams u_Fade;
};

#ifdef GLSL_VS
ia_out(GLSL_POSITION)           vec2 ia_VertexPosition;
#endif
#if defined(GLSL_VS) || defined(GLSL_PS)
vs_out vec4 vs_FadeColor;
#endif

#if defined(GLSL_VS)


void main(void)
{
    vs_FadeColor = u_Fade.color;
    gl_Position = vec4(ia_VertexPosition, 0.0, 1.0);
}

#elif defined(GLSL_PS)

ps_out(0) vec4 ps_FragColor;

void main()
{
    ps_FragColor = vs_FadeColor;
}

#endif


