#ifndef __atomic_Graphics_Shader__
#define __atomic_Graphics_Shader__
namespace atomic {


class ShaderGBuffer : public ProgramObject
{
typedef ProgramObject super;
private:
    VertexShader m_vsh;
    FragmentShader m_fsh;

    GLuint m_block_instance_position;

public:
    bool initialize();

    void setInstancePositionBinding(int index) { setUniformBlockBinding(m_block_instance_position, index); }
};


class ShaderDeferred : public ProgramObject
{
typedef ProgramObject super;
private:
    VertexShader m_vsh;
    FragmentShader m_fsh;

    GLuint m_loc_color_buffer;
    GLuint m_loc_glow_buffer;
    GLuint m_loc_normal_buffer;
    GLuint m_loc_position_buffer;
    GLuint m_loc_depth_buffer;

    GLuint m_loc_aspect_ratio;
    GLuint m_loc_texcoord_scale;

    GLuint m_block_light_position;

public:
    bool initialize();

    void setColorBuffer(int32 v)    { setUniform1i(m_loc_color_buffer, v); }
    void setGlowBuffer(int32 v)     { setUniform1i(m_loc_glow_buffer, v); }
    void setNormalBuffer(int32 v)   { setUniform1i(m_loc_normal_buffer, v); }
    void setPositionBuffer(int32 v) { setUniform1i(m_loc_position_buffer, v); }
    void setDepthBuffer(int32 v)    { setUniform1i(m_loc_depth_buffer, v); }
    void setAspectRatio(float32 v)  { setUniform1f(m_loc_aspect_ratio, v); }
    void setTexcoordScale(float32 v1, float32 v2) { setUniform2f(m_loc_texcoord_scale, v1, v2); }

    void setLightPositionBinding(int index) { setUniformBlockBinding(m_block_light_position, index); }
};


class ShaderOutput : public ProgramObject
{
typedef ProgramObject super;
private:
    VertexShader m_vsh;
    FragmentShader m_fsh;

public:
    bool initialize();
};

} // namespace atomic
#endif // __atomic_Graphics_Shader__
