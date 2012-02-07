#ifndef __atomic_RenderStates__
#define __atomic_RenderStates__
#ifndef GLSL
namespace atomic {
#endif // GLSL

struct
#ifndef GLSL
    __declspec(align(16))
#endif // GLSL
    RenderStates
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


struct
#ifndef GLSL
    __declspec(align(16))
#endif // GLSL
    FXAAParams
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

struct
#ifndef GLSL
    __declspec(align(16))
#endif // GLSL
    FadeParams
{
    vec4 color;
};

struct
#ifndef GLSL
    __declspec(align(16))
#endif // GLSL
    DebugShowBufferParams
{
    vec2 bottom_left; // スクリーン座標、-1.0～1.0
    vec2 upper_right; // スクリーン座標、-1.0～1.0
    vec2 color_range; // この範囲を 0.0～1.0 に clamp する
};


#ifndef GLSL
} // namespace atomic
#endif // GLSL
#endif // __atomic_RenderStates__
