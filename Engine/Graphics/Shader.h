#ifndef atm_Engine_Graphics_Shader_h
#define atm_Engine_Graphics_Shader_h

#include "shader/Semantics.h"
#include "shader/RenderStates.h"

namespace atm {

class AtomicShader
{
typedef ShaderProgram super;
public:
    AtomicShader();
    ~AtomicShader();
    void release();

    void clearShaders();
    bool createShaders(const char* filename);

    int32 getUniformBlockIndex(const char *name) const;
    void setUniformBlock(GLuint uniformBlockIndex, GLuint uniformBindingIndex, Buffer *buffer);
    void bind();
    void unbind();
    void assign(i3d::DeviceContext *dc);

protected:
    ShaderProgram   *m_shader;
    int32 m_loc_renderstates;

#ifdef atm_enable_ShaderLiveEdit
public:
    bool needsRecompile();
    bool recompile();

private:
    stl::string m_glsl_filename;
    Poco::Timestamp m_timestamp;
#endif // atm_enable_ShaderLiveEdit

};


} // namespace atm
#endif // atm_Engine_Graphics_Shader_h
