#version 410 compatibility

uniform sampler2D u_ColorBuffer;
uniform float u_RcpScreenWidth;
uniform float u_RcpScreenHeight;
uniform vec2 u_TexcoordMin;
uniform vec2 u_TexcoordMax;
in vec2 v_Texcoord;

layout(location=0) out vec4 o_FragColor;

const float Weight[8] = float[8](0.1284, 0.1221, 0.1051, 0.0819, 0.0577, 0.0368, 0.0212, 0.0111);


subroutine vec4 passType();
subroutine uniform passType u_Pass;

subroutine(passType)
vec4 pickup()
{
    vec4 color = texture(u_ColorBuffer, v_Texcoord);
    return pow(max(color-vec4(0.5), vec4(0.0))*2.0, vec4(2.0));
}

subroutine(passType)
vec4 horizontalBlur()
{
    vec4 color;
    for(int i=0; i<8; ++i) {
        vec2 gap = vec2(float(i)*u_RcpScreenWidth, 0.0);
        color += (texture(u_ColorBuffer, clamp(v_Texcoord+gap, u_TexcoordMin, u_TexcoordMax)) + 
                  texture(u_ColorBuffer, clamp(v_Texcoord-gap, u_TexcoordMin, u_TexcoordMax))) * Weight[i];
    }
    return color;
}

subroutine(passType)
vec4 verticalBlur()
{
    vec4 color;
    for(int i=0; i<8; ++i) {
        vec2 gap = vec2(0.0, float(i)*u_RcpScreenHeight);
        color += (texture(u_ColorBuffer, clamp(v_Texcoord+gap, u_TexcoordMin, u_TexcoordMax)) + 
                  texture(u_ColorBuffer, clamp(v_Texcoord-gap, u_TexcoordMin, u_TexcoordMax))) * Weight[i];
    }
    return color;
}

subroutine(passType)
vec4 composite()
{
    return texture(u_ColorBuffer, v_Texcoord) * 0.75;
}


void main()
{
    o_FragColor = u_Pass();
}
