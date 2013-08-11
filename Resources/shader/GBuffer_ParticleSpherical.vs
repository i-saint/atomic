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
layout(location=1) in             vec3 ia_VertexNormal;
layout(location=8) in  vec4 ia_InstancePosition;
layout(location=10) in     vec4 ia_InstanceColor;
layout(location=15) in      vec4 ia_InstanceGlow;
layout(location=14) in     vec4 ia_InstanceParam; 

out vec4 vs_InstancePosition;
out vec4 vs_VertexPosition;      
out vec4 vs_VertexNormal;        
out vec4 vs_VertexColor;         
out vec4 vs_Glow;
out float vs_Scale;

void main()
{
    float scale = ia_InstanceParam.x;
    vec4 vert = vec4((ia_VertexPosition.xyz*scale)+ia_InstancePosition.xyz, 1.0);

    vs_Glow = ia_InstanceGlow;
    vs_Scale = scale;

    vs_VertexPosition = vert;
    vs_VertexNormal = vec4(ia_VertexNormal, 0.04);
    vs_VertexColor = ia_InstanceColor;
    vs_InstancePosition = ia_InstancePosition;
    gl_Position = u_RS.ModelViewProjectionMatrix * vert;
}

