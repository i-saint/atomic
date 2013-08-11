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
layout(location=1) in             vec4 ia_VertexNormal;
layout(location=8) in  vec4 ia_InstancePosition;
layout(location=10) in     vec4 ia_InstanceColor;
layout(location=14) in     vec4 ia_InstanceParam;

out vec4 vs_LightPosition;
out vec4 vs_LightPositionMVP;
out vec4 vs_LightColor;
out vec4 vs_VertexPositionMVP;
out float vs_LightRange;
out float vs_RcpLightRange;

void main()
{
    vs_LightPosition = ia_InstancePosition;
    vs_LightPosition.w = 1.0;
    vs_LightRange = ia_InstanceParam.x;
    vs_RcpLightRange = ia_InstanceParam.y;

    vec4 scaled_position = ia_VertexPosition * vec4(vs_LightRange, vs_LightRange, vs_LightRange, 0.0);

    vs_LightColor = ia_InstanceColor;
    vs_LightPositionMVP = u_RS.ModelViewProjectionMatrix * vs_LightPosition;
    vs_VertexPositionMVP= u_RS.ModelViewProjectionMatrix * (scaled_position+vs_LightPosition);
    gl_Position = vs_VertexPositionMVP;
}

