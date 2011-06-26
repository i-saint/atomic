#version 410 compatibility

layout(std140) uniform FractionData
{
    vec4 fraction_pos[1024];
};

out vec3 normal;
out vec4 color;

void main(void)
{
    vec4 fractionPos = fraction_pos[gl_InstanceID];
    vec3 position = vec3(gl_ModelViewMatrix * (gl_Vertex));
    normal = normalize(gl_NormalMatrix * gl_Normal);
    vec3 light = normalize(gl_LightSource[0].position.xyz - position);
    vec3 view = normalize(position);
    float diffuse = dot(light, normal);

    color = gl_FrontLightProduct[0].ambient;
    vec3 halfway = normalize(light - view);
    float specular = pow(max(dot(normal, halfway), 0.0), gl_FrontMaterial.shininess);
    color += gl_FrontLightProduct[0].diffuse * diffuse
                  +  gl_FrontLightProduct[0].specular * specular;

    gl_Position = gl_ModelViewProjectionMatrix * (gl_Vertex+fractionPos);
    //gl_TexCoord[1] = gl_TextureMatrix[1] * gl_Vertex;
}
