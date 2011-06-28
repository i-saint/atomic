#version 410 compatibility

uniform sampler2D ColorBuffer;
in vec2 Texcoord;

layout(location=0) out vec4 FragColor;


void main()
{
    FragColor = texture2D(ColorBuffer, Texcoord);
}
