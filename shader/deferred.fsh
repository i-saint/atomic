#version 410 compatibility

uniform sampler2D u_ColorBuffer;
uniform sampler2D u_NormalBuffer;
uniform sampler2D u_PositionBuffer;
uniform float u_AspectRatio;
uniform vec2 u_TexcoordScale;

in vec4 v_LightPosition;
in vec4 v_LightPositionMVP;
in vec4 v_LightColor;
in vec4 v_VertexPositionMVP;

layout(location=0) out vec4 o_FragColor;



void main()
{
    vec2 coord;
    coord.x = (1.0 + (v_VertexPositionMVP.x/v_VertexPositionMVP.w))*0.5;
    coord.y = (1.0 + (v_VertexPositionMVP.y/v_VertexPositionMVP.w))*0.5 / u_AspectRatio;
    coord *= u_TexcoordScale;

    vec4 frag_position = texture(u_PositionBuffer, coord);

    vec3 Albedo = texture(u_ColorBuffer, coord).xyz;
    vec3 Ambient = normalize(frag_position.xyz)*0.05;
    vec3 Normal = texture(u_NormalBuffer, coord).xyz;
    vec3 L = normalize(v_LightPosition.xyz - frag_position.xyz);
    vec4 color;
    color.xyz = v_LightColor.xyz * (
                    Ambient +
                    Albedo * max(dot(Normal,L), 0.0)
                 );

    float strength = max(125.0-length(frag_position.xyz-v_LightPosition.xyz), 0.0)/125.0;
    color.rgb *= pow(strength, 0.7);

    o_FragColor = color;

    o_FragColor.w = 1.0;
}
