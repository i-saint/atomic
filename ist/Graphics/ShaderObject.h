#ifndef __ist_Graphics_ShaderObject__
#define __ist_Graphics_ShaderObject__

#include "types.h"
#include "GraphicsResource.h"

namespace ist {
namespace graphics {

template<size_t ShaderType>
class ShaderObject : public GraphicsResource
{
public:
    enum {
        SHADER_TYPE = ShaderType,
    };
private:
    GLuint m_handle;

public:
    ShaderObject();
    ~ShaderObject();

    bool initialize();
    bool initialize(const char *src, int length);
    void finalize();
    bool  compile(const char *src, int length);

    GLuint getHandle() const;
};

typedef ShaderObject<GL_VERTEX_SHADER> VertexShader;
typedef ShaderObject<GL_FRAGMENT_SHADER> FragmentShader;
typedef ShaderObject<GL_GEOMETRY_SHADER> GeometryShader;

class ProgramObject : public GraphicsResource
{
private:
    GLuint m_handle;

private:

public:
    ProgramObject();
    ~ProgramObject();

    bool initialize();
    void finalize();

    bool link(VertexShader *vsh, FragmentShader *fsh, GeometryShader *gsh=NULL);
    void bind();
    void unbind();


    GLint getUniformLocation(const char *name) const;
    GLint getAttribLocation(const char *name) const;

    GLint getUniformBlockIndex(const char *name) const;
    void setUniformBlock(GLuint uniformBlockIndex, GLuint uniformBindingIndex, GLuint uniformBufferHandle);

    // uniform variable
    // int
    void setUniform1i(GLint al, GLint v);
    void setUniform2i(GLint al, const ivec2& v);
    void setUniform3i(GLint al, const ivec3& v);
    void setUniform4i(GLint al, const ivec4& v);
    // float
    void setUniform1f(GLint al, GLfloat v);
    void setUniform2f(GLint al, const vec2& v);
    void setUniform3f(GLint al, const vec3& v);
    void setUniform4f(GLint al, const vec4& v);
    // int array
    void setUniform1iv(GLint al, GLuint count, const GLint *v);
    void setUniform2iv(GLint al, GLuint count, const GLint *v);
    void setUniform3iv(GLint al, GLuint count, const GLint *v);
    void setUniform4iv(GLint al, GLuint count, const GLint *v);
    // float array
    void setUniform1fv(GLint al, GLuint count, const GLfloat *v);
    void setUniform2fv(GLint al, GLuint count, const GLfloat *v);
    void setUniform3fv(GLint al, GLuint count, const GLfloat *v);
    void setUniform4fv(GLint al, GLuint count, const GLfloat *v);
    // matrix
    void setUniformMatrix2fv(GLint al, GLuint count, GLboolean transpose, const GLfloat *v);
    void setUniformMatrix3fv(GLint al, GLuint count, GLboolean transpose, const GLfloat *v);
    void setUniformMatrix4fv(GLint al, GLuint count, GLboolean transpose, const GLfloat *v);

    // attribute variable
    // float
    void setVertexAttrib1f(GLint al, GLfloat v0);
    void setVertexAttrib2f(GLint al, GLfloat v0, GLfloat v1);
    void setVertexAttrib3f(GLint al, GLfloat v0, GLfloat v1, GLfloat v2);
    void setVertexAttrib4f(GLint al, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3);
    // float array
    void setVertexAttrib1fv(GLint al, const GLfloat *v);
    void setVertexAttrib2fv(GLint al, const GLfloat *v);
    void setVertexAttrib3fv(GLint al, const GLfloat *v);
    void setVertexAttrib4fv(GLint al, const GLfloat *v);

    // subroutine
    GLuint getSubroutineIndexV(const char *name);
    GLuint getSubroutineIndexG(const char *name);
    GLuint getSubroutineIndexF(const char *name);
    void setSubroutineV(GLsizei count, GLuint *indices);
    void setSubroutineG(GLsizei count, GLuint *indices);
    void setSubroutineF(GLsizei count, GLuint *indices);
};


} // namespace graphics
} // namespace ist
#endif // __ist_Graphics_ShaderObject__
