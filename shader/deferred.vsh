#version 410 compatibility

layout(std140) uniform LightData
{
    vec4 light_pos_array[1024];
};

out vec4 light_pos;
out vec4 light_screen_pos;
out vec4 light_color;
out vec4 screen_pos;


void main(void)
{
    light_pos = light_pos_array[gl_InstanceID]*1.1;

    light_color = normalize(light_pos);
    screen_pos = gl_ModelViewProjectionMatrix * (gl_Vertex+light_pos);
    light_screen_pos = gl_ModelViewProjectionMatrix * light_pos;
    gl_Position = screen_pos;
}
