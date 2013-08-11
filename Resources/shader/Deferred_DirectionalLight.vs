#version 330 core

    

struct  RenderStates
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

struct  FXAAParams
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

struct  FadeParams
{
    vec4 Color;
};

struct  FillParams
{
    vec4 Color;
};

struct  MultiresolutionParams
{
    ivec4 Level;
    vec4 Threshold;
};

struct  DebugShowBufferParams
{
    vec2 BottomLeft; 
    vec2 UpperRight; 
    vec2 ColorRange; 
};

    layout(std140) uniform render_states
    {
        RenderStates u_RS;
    };

    uniform sampler2D u_ParamBuffer;

    
    

    

float SmoothInterpolation2(float v)
{
    return v < 0.5 ? 2.0*v*v : -1.0+4.0*v-2*v*v;
}

float SmoothInterpolation(float v, int n)
{
    return v < 0.5 ? pow(2.0*v, n) : 1.0 - pow(2.0-2.0*v, n);
}

layout(location=0) in           vec4 ia_VertexPosition;
layout(location=3) in          vec2 ia_VertexTexcoord0;
layout(location=13) in vec4 ia_InstanceDirection;
layout(location=10) in     vec4 ia_InstanceColor;
layout(location=11) in   vec4 ia_InstanceAmbient;

out vec2 vs_Texcoord;
out vec4 vs_LightDirection;
out vec4 vs_LightColor;
out vec4 vs_LightAmbient;
out vec4 vs_VertexPositionMVP;

void main()
{
    vs_Texcoord = ia_VertexTexcoord0*u_RS.ScreenTexcoord;

    vs_LightDirection   = ia_InstanceDirection;
    vs_LightColor       = ia_InstanceColor;
    vs_LightAmbient     = ia_InstanceAmbient;
    gl_Position         = ia_VertexPosition;
}

