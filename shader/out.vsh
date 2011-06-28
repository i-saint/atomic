#version 410 compatibility

out vec2 texcoord;

void main(void)
{
    texcoord = gl_MultiTexCoord0.xy;
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
}
