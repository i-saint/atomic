#ifndef atomic_Graphics_Shader_h
#define atomic_Graphics_Shader_h

#include "shader/Semantics.h"
#include "shader/RenderStates.h"

namespace atomic {

class AtomicShader
{
typedef ShaderProgram super;
protected:
    ShaderProgram   *m_shader;
    VertexShader    *m_vs;
    PixelShader     *m_ps;
    GeometryShader  *m_gs;
    int32 m_loc_renderstates;

public:
    AtomicShader();
    ~AtomicShader();
    void release();

    bool loadFromMemory(const char* src);

    GLint getUniformBlockIndex(const char *name) const;
    void setUniformBlock(GLuint uniformBlockIndex, GLuint uniformBindingIndex, GLuint uniformBufferHandle);
    void bind();
    void unbind();
    void assign(i3d::DeviceContext *dc);
};


} // namespace atomic
#endif // atomic_Graphics_Shader_h
