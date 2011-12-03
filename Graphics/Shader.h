#ifndef __atomic_Graphics_Shader_h__
#define __atomic_Graphics_Shader_h__

namespace atomic {

#include "shader/semantics.h"

class ShaderGBuffer : public ProgramObject
{
typedef ProgramObject super;
private:
    VertexShader m_vsh;
    FragmentShader m_fsh;

public:
    bool initialize();
};

class ShaderGBuffer_Octahedron : public ProgramObject
{
typedef ProgramObject super;
private:
    VertexShader m_vsh;
    FragmentShader m_fsh;

public:
    bool initialize();
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

    GLuint m_loc_rcp_aspect_ratio;
    GLuint m_loc_texcoord_scale;

public:
    bool initialize();

    void setColorBuffer(int32 v)        { setUniform1i(m_loc_color_buffer, v); }
    void setGlowBuffer(int32 v)         { setUniform1i(m_loc_glow_buffer, v); }
    void setNormalBuffer(int32 v)       { setUniform1i(m_loc_normal_buffer, v); }
    void setPositionBuffer(int32 v)     { setUniform1i(m_loc_position_buffer, v); }
    void setDepthBuffer(int32 v)        { setUniform1i(m_loc_depth_buffer, v); }
    void setRcpAspectRatio(float32 v)   { setUniform1f(m_loc_rcp_aspect_ratio, v); }
    void setTexcoordScale(const vec2& v) { setUniform2f(m_loc_texcoord_scale, v); }
};


class ShaderBloom : public ProgramObject
{
typedef ProgramObject super;
private:
    VertexShader m_vsh;
    FragmentShader m_fsh;

    GLuint m_loc_color_buffer;
    GLuint m_loc_rcp_screen_width;
    GLuint m_loc_rcp_screen_height;
    GLuint m_loc_texcoord_min;
    GLuint m_loc_texcoord_max;

    GLuint m_sub_pickup;
    GLuint m_sub_hblur;
    GLuint m_sub_vblur;
    GLuint m_sub_composite;

public:
    bool initialize();
    void setColorBuffer(int32 v)        { setUniform1i(m_loc_color_buffer, v); }
    void setRcpScreenWidth(float32 v)   { setUniform1f(m_loc_rcp_screen_width, v); }
    void setRcpScreenHeight(float32 v)  { setUniform1f(m_loc_rcp_screen_height, v); }
    void setTexcoordMin(const vec2& v)  { setUniform2f(m_loc_texcoord_min, v); }
    void setTexcoordMax(const vec2& v)  { setUniform2f(m_loc_texcoord_max, v); }

    void switchToPickupPass()           { setSubroutineF(1, &m_sub_pickup); }
    void switchToHorizontalBlurPass()   { setSubroutineF(1, &m_sub_hblur); }
    void switchToVerticalBlurPass()     { setSubroutineF(1, &m_sub_vblur); }
    void switchToCompositePass()        { setSubroutineF(1, &m_sub_composite); }
};

class ShaderOutput : public ProgramObject
{
typedef ProgramObject super;
private:
    VertexShader m_vsh;
    FragmentShader m_fsh;

    GLuint m_loc_color_buffer;

public:
    bool initialize();
    void setColorBuffer(int32 v)    { setUniform1i(m_loc_color_buffer, v); }
};

} // namespace atomic
#endif // __atomic_Graphics_Shader_h__
