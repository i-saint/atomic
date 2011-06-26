#version 410 compatibility

in vec3 normal;
in vec4 color;

void main()
{
    //color.rgb = normal.xyz;
    gl_FragColor = color;
}
