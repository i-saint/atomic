#version 410 compatibility

uniform sampler2D u_ColorBuffer;
in vec2 v_Texcoord;

layout(location=0) out vec4 o_FragColor;


void main()
{
    o_FragColor = texture(u_ColorBuffer, v_Texcoord);
}
