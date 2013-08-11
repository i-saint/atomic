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

const vec2 u_ScreenSize = vec2(512.0, 256.0);
const vec2 u_RcpScreenSize = vec2(1.0,1.0)/u_ScreenSize;
const vec2 u_HalfPixel = u_RcpScreenSize*0.5;

layout(location=0) in           vec2 ia_VertexPosition;
layout(location=3) in          vec2 ia_VertexTexcoord0;
layout(location=4) in          vec2 ia_VertexTexcoord1;
layout(location=5) in          vec2 ia_VertexTexcoord2;
layout(location=6) in          vec2 ia_VertexTexcoord3;

out vec2 vs_Texcoord;
out vec2 vs_TexcoordMin;
out vec2 vs_TexcoordMax;
out vec2 vs_Texcoords[4];

void main()
{
    vs_TexcoordMin = ia_VertexTexcoord1+u_HalfPixel;
    vs_TexcoordMax = ia_VertexTexcoord2-u_HalfPixel;
    vs_TexcoordMax.y *= u_RS.RcpAspectRatio;
    vs_Texcoord = ia_VertexTexcoord0;
    gl_Position = vec4(ia_VertexPosition, 0.0, 1.0);
}

