#version 410 compatibility

out vec2 v_Texcoord;

void main(void)
{
    v_Texcoord = gl_MultiTexCoord0.xy;
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
}
