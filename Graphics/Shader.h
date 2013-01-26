#ifndef atomic_Graphics_Shader_h
#define atomic_Graphics_Shader_h

#include "shader/Semantics.h"
#include "shader/RenderStates.h"

namespace atomic {

class AtomicShader
{
typedef ShaderProgram super;
public:
    AtomicShader();
    ~AtomicShader();
    void release();

    void clearShaders();
    bool createShaders(const char* filename);

    GLint getUniformBlockIndex(const char *name) const;
    void setUniformBlock(GLuint uniformBlockIndex, GLuint uniformBindingIndex, Buffer *buffer);
    void bind();
    void unbind();
    void assign(i3d::DeviceContext *dc);

protected:
    ShaderProgram   *m_shader;
    int32 m_loc_renderstates;

#ifdef atomic_enable_shader_live_edit
public:
    bool needsRecompile();
    bool recompile();

private:
    stl::string m_glsl_filename;
    Poco::Timestamp m_timestamp;
#endif // atomic_enable_shader_live_edit

};


} // namespace atomic
#endif // atomic_Graphics_Shader_h
