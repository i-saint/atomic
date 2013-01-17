#ifndef atomic_RenderStates_h
#define atomic_RenderStates_h

#ifdef GLSL
    #define ALIGN16 
#else // GLSL
    #define ALIGN16 istAlign(16)
#endif // GLSL

#ifndef GLSL
namespace atomic {
#endif // GLSL

struct ALIGN16 RenderStates
{
    mat4 ModelViewProjectionMatrix;
    vec4 CameraPosition;
    vec4 CameraDirection;
    vec4 Color;

    vec2 ScreenTexcoord;
    vec2 ScreenSize;
    vec2 RcpScreenSize;
    float AspectRatio;
    float RcpAspectRatio;
    float Frame;
    int Level;
    int ShowMultiresolution;
};


struct ALIGN16 FXAAParams
{
    vec4 fxaaConsolePosPos;
    vec4 fxaaConsoleRcpFrameOpt;
    vec4 fxaaConsoleRcpFrameOpt2;
    vec4 fxaaConsole360RcpFrameOpt2;
    vec4 fxaaConsole360ConstDir;
    vec2 fxaaQualityRcpFrame;
    float fxaaQualitySubpix;
    float fxaaQualityEdgeThreshold;
    float fxaaQualityEdgeThresholdMin;
    float fxaaConsoleEdgeSharpness;
    float fxaaConsoleEdgeThreshold;
    float fxaaConsoleEdgeThresholdMin;
};

struct ALIGN16 FadeParams
{
    vec4 Color;
};

struct ALIGN16 FillParams
{
    vec4 Color;
};

struct ALIGN16 MultiresolutionParams
{
    ivec4 Level;
    vec4 Threshold;
};

struct ALIGN16 DebugShowBufferParams
{
    vec2 BottomLeft; // スクリーン座標、-1.0～1.0
    vec2 UpperRight; // スクリーン座標、-1.0～1.0
    vec2 ColorRange; // この範囲を 0.0～1.0 に clamp する
};


#ifndef GLSL
} // namespace atomic
#endif // GLSL
#undef ALIGN16
#endif // atomic_RenderStates_h
