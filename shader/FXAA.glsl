#include "Common.h"

#define FXAA_PC 1
#define FXAA_GLSL_130 1
#define FXAA_QUALITY__PRESET 12
#include "Fxaa3_11.h"


layout(std140) uniform fxaa_params
{
    FXAAParams u_FXAA;
};

#ifdef GLSL_VS
ia_out(GLSL_POSITION)           vec2 ia_VertexPosition;
ia_out(GLSL_TEXCOORD0)          vec2 ia_VertexTexcoord0;
#endif
#if defined(GLSL_VS) || defined(GLSL_PS)
vs_out vec2 vs_Texcoord;
#endif

#if defined(GLSL_VS)

void main(void)
{
    vs_Texcoord = ia_VertexTexcoord0 * u_RS.ScreenTexcoord;
    gl_Position = vec4(ia_VertexPosition, 0.0, 1.0);
}

#elif defined(GLSL_PS)

ps_out(0) vec4 ps_FragColor;

void main()
{
    ps_FragColor = FxaaPixelShader(
        vs_Texcoord,
        u_FXAA.fxaaConsolePosPos,
        u_ColorBuffer,
        u_ColorBuffer,
        u_ColorBuffer,
        u_FXAA.fxaaQualityRcpFrame,
        u_FXAA.fxaaConsoleRcpFrameOpt,
        u_FXAA.fxaaConsoleRcpFrameOpt2,
        u_FXAA.fxaaConsole360RcpFrameOpt2,
        u_FXAA.fxaaQualitySubpix,
        u_FXAA.fxaaQualityEdgeThreshold,
        u_FXAA.fxaaQualityEdgeThresholdMin,
        u_FXAA.fxaaConsoleEdgeSharpness,
        u_FXAA.fxaaConsoleEdgeThreshold,
        u_FXAA.fxaaConsoleEdgeThresholdMin,
        u_FXAA.fxaaConsole360ConstDir
    );
}

#endif
