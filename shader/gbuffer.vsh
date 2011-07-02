#version 410 compatibility

layout(std140) uniform InstancePosition
{
    vec4 instance_pos[1024];
};

out vec3 VertexPosition;
out vec3 VertexNormal;
out vec4 VertexColor;

void main(void)
{
    vec4 fractionPos = instance_pos[gl_InstanceID];
    vec4 vert = gl_Vertex+fractionPos;

    VertexPosition = (gl_Vertex+fractionPos).xyz;
    VertexNormal = gl_Normal;
    VertexColor = vec4(0.6, 0.6, 0.6, 1.0);
    gl_Position = gl_ModelViewProjectionMatrix * vert;
}
