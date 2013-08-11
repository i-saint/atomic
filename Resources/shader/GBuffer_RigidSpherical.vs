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
layout(location=8) in  vec3 ia_InstancePosition;
layout(location=9) in    vec4 ia_InstanceNormal;
layout(location=14) in     int  ia_InstanceID;

out vec4 vs_PSetPosition;
out vec4 vs_InstancePosition;
out vec4 vs_InstanceParams;      
out vec4 vs_VertexPosition;      
out vec4 vs_VertexNormal;        
out vec4 vs_VertexColor;         
out vec4 vs_Glow;
out vec4 vs_Flash;

const float scale = 1.2;

void main()
{
    vec4 ia_InstanceColor   = texelFetch(u_ParamBuffer, ivec2(0, ia_InstanceID), 0);
    vec4 ia_InstanceGlow    = texelFetch(u_ParamBuffer, ivec2(1, ia_InstanceID), 0);
    vec4 ia_InstanceFlash   = texelFetch(u_ParamBuffer, ivec2(2, ia_InstanceID), 0);
    vs_InstanceParams = texelFetch(u_ParamBuffer, ivec2(3, ia_InstanceID), 0);

    mat4 trans;
    trans[0] = texelFetch(u_ParamBuffer, ivec2(4, ia_InstanceID), 0);
    trans[1] = texelFetch(u_ParamBuffer, ivec2(5, ia_InstanceID), 0);
    trans[2] = texelFetch(u_ParamBuffer, ivec2(6, ia_InstanceID), 0);
    trans[3] = texelFetch(u_ParamBuffer, ivec2(7, ia_InstanceID), 0);
    vs_PSetPosition = trans[3];

    vec4 instancePos = trans * vec4(ia_InstancePosition, 1.0);
    vec4 vert = ia_VertexPosition*vs_InstanceParams.z*scale + instancePos;
    vert.w = 1.0;

    float dif = ia_InstanceNormal.w;
    float dif2 = dif*dif;
    float dif3 = dif2*dif;
    float ndif = 1.0-dif;
    vs_Glow = ia_InstanceGlow * (ndif*ndif);
    vs_Flash = ia_InstanceFlash;

    vs_VertexPosition = vec4(vert.xyz, 1.0);
    vs_VertexNormal = vec4(ia_VertexNormal, 0.04);
    vs_VertexColor = vec4(ia_InstanceColor.rgb*dif3, ia_InstanceColor.a);
    vs_InstancePosition = instancePos;
    gl_Position = u_RS.ModelViewProjectionMatrix * vert;
}

