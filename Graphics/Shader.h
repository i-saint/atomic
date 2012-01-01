#ifndef __atomic_Graphics_Shader_h__
#define __atomic_Graphics_Shader_h__

#include "shader/Semantics.glslh"
#include "shader/RenderStates.glslh"

namespace atomic {

class AtomicShader : public ShaderProgram
{
typedef ShaderProgram super;
protected:
    //ProgramObject   m_program;
    VertexShader   m_vsh;
    PixelShader    m_fsh;
    int32 m_loc_renderstates;
    int32 m_rs_binding;

public:
    AtomicShader();
    bool initialize();
    void finalize();
    virtual bool loadFromMemory(const char* src);

    void bind();
};


} // namespace atomic
#endif // __atomic_Graphics_Shader_h__
