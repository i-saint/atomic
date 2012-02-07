#ifndef __atomic_RenderStates__
#define __atomic_RenderStates__

#ifdef GLSL
    #define ALIGN16 
#else // GLSL
    #define ALIGN16 __declspec(align(16))
#endif // GLSL

#ifndef GLSL
namespace atomic {
#endif // GLSL

struct ALIGN16 RenderStates
{
    mat4 ModelViewProjectionMatrix;
    vec4 CameraPosition;

    vec2 ScreenTexcoord;
    vec2 ScreenSize;
    vec2 RcpScreenSize;
    float AspectRatio;
    float RcpAspectRatio;
    int Level;
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
    vec4 color;
};

struct ALIGN16 FillParams
{
    vec4 color;
};

struct ALIGN16 DebugShowBufferParams
{
    vec2 bottom_left; // スクリーン座標、-1.0～1.0
    vec2 upper_right; // スクリーン座標、-1.0～1.0
    vec2 color_range; // この範囲を 0.0～1.0 に clamp する
};


#ifndef GLSL
} // namespace atomic
#endif // GLSL
#undef ALIGN16
#endif // __atomic_RenderStates__
