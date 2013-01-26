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

    bool createShaders(const char* filename);

    GLint getUniformBlockIndex(const char *name) const;
    void setUniformBlock(GLuint uniformBlockIndex, GLuint uniformBindingIndex, Buffer *buffer);
    void bind();
    void unbind();
    void assign(i3d::DeviceContext *dc);

protected:
    ShaderProgram   *m_shader;
    VertexShader    *m_vs;
    PixelShader     *m_ps;
    GeometryShader  *m_gs;
    int32 m_loc_renderstates;
#ifdef atomic_enable_shader_live_edit
#endif // atomic_enable_shader_live_edit

};


} // namespace atomic
#endif // atomic_Graphics_Shader_h
