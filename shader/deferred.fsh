#version 410 compatibility

uniform sampler2D ColorBuffer;
uniform sampler2D NormalBuffer;
uniform sampler2D PositionBuffer;
uniform float AspectRatio;
uniform vec2 TexcoordScale;

in vec4 light_pos;
in vec4 light_screen_pos;
in vec4 light_color;
in vec4 screen_pos;

layout(location=0) out vec4 FragColor;



void main()
{
    vec2 coord;
    coord.x = (1.0 + (screen_pos.x/screen_pos.w))*0.5;
    coord.y = (1.0 + (screen_pos.y/screen_pos.w))*0.5 / AspectRatio;
    coord *= TexcoordScale;

    vec4 frag_position = texture(PositionBuffer, coord);

    vec3 Albedo = texture(ColorBuffer, coord).xyz;
    vec3 Ambient = normalize(frag_position.xyz)*0.05;
    vec3 Normal = texture(NormalBuffer, coord).xyz;
    vec3 L = normalize(light_pos.xyz - frag_position.xyz);
    vec4 color;
    color.xyz = light_color.xyz * (
                    Ambient +
                    Albedo * max(dot(Normal,L), 0.0)
                 );

    float strength = max(125.0-length(frag_position.xyz-light_pos.xyz), 0.0)/125.0;
    color.rgb *= pow(strength, 0.7);

    FragColor = color;

    FragColor.w = 1.0;
}
