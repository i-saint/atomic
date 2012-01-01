#ifndef __atomic_Graphics_Shader_h__
#define __atomic_Graphics_Shader_h__

#include "shader/Semantics.glslh"
#include "shader/RenderStates.glslh"

namespace atomic {

class AtomicShader
{
typedef ShaderProgram super;
protected:
    ShaderProgram   *m_shader;
    VertexShader    *m_vs;
    PixelShader     *m_ps;
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
};


} // namespace atomic
#endif // __atomic_Graphics_Shader_h__
