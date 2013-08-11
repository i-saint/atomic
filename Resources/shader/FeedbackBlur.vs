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

layout(location=0) in            vec4 ia_VertexPosition;
layout(location=1) in              vec4 ia_VertexNormal;
layout(location=8) in mat4 ia_InstanceTransform;
layout(location=12) in     vec4 ia_InstanceParams; 

out vec4 vs_Params;
out vec4 vs_Outer;
out vec4 vs_Position;

void main()
{
    float strength = ia_InstanceParams.y;
    vec4 instance_pos = ia_InstanceTransform[3];
    vec4 vertex_pos = ia_InstanceTransform * ia_VertexPosition;
    vec3 dir = normalize(vertex_pos.xyz-instance_pos.xyz);
    vec4 outer_pos = vec4(vertex_pos.xyz + dir*strength, 1.0f);

    vs_Params = ia_InstanceParams;
    vs_Outer = u_RS.ModelViewProjectionMatrix * outer_pos;
    vs_Position = u_RS.ModelViewProjectionMatrix * vertex_pos;
    gl_Position = vs_Position;
}

