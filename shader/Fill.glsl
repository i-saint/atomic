#version 330 core
#pragma include("Common.h")

layout(std140) uniform fill_params
{
    FillParams u_Fill;
};

#ifdef GLSL_VS
ia_out(GLSL_POSITION)           vec2 ia_VertexPosition;
#endif

#if defined(GLSL_VS)

void main()
{
    gl_Position = vec4(ia_VertexPosition, 0.0, 1.0);
}

#elif defined(GLSL_PS)

ps_out(0) vec4 ps_FragColor;

void main()
{
    ps_FragColor = u_Fill.Color;
}

#endif
