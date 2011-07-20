#version 410 compatibility

out vec2 Texcoord;

void main(void)
{
    Texcoord = gl_MultiTexCoord0.xy;
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;

    gl_Position = ftransform();
    gl_Position = sign( gl_Position );
     
}
