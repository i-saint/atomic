#version 410 compatibility

uniform sampler2D screen;
in vec2 texcoord;

layout(location=0, index=0) out vec4 FragColor;


void main()
{
    FragColor = texture2D(screen, texcoord);
}
