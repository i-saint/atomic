#ifndef __ist_Graphics_ShaderObject__
#define __ist_Graphics_ShaderObject__

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

    GLuint getHandle() const { return m_handle; }

    bool initialize(const char *src, int length);
    void finalize();
};

typedef ShaderObject<GL_VERTEX_SHADER> VertexShader;
typedef ShaderObject<GL_FRAGMENT_SHADER> FragmentShader;
typedef ShaderObject<GL_GEOMETRY_SHADER> GeometryShader;

class ProgramObject : public GraphicsResource
{
private:
    GLuint m_handle;

private:
    void attachVertexShader(VertexShader *sh);
    void attachGeometryShader(GeometryShader *sh);
    void attachFragmentShader(FragmentShader *sh);
    bool link();

public:
    ProgramObject();
    ~ProgramObject();

    bool operator!() const { return !m_handle; }

    bool initialize(VertexShader *vsh, GeometryShader *gsh, FragmentShader *fsh);
    void finalize();

    void bind() const;
    void unbind() const;


    GLint getUniformLocation(const char *name);
    GLint getAttribLocation(const char *name);

    GLint getUniformBlockIndex(const char *name);
    void setUniformBlockBinding(GLuint uniformBlockIndex, GLuint uniformBufferHandle);

    // uniform variable
    // int
    void setUniform1i(GLint al, GLint v0);
    void setUniform2i(GLint al, GLint v0, GLint v1);
    void setUniform3i(GLint al, GLint v0, GLint v1, GLint v2);
    void setUniform4i(GLint al, GLint v0, GLint v1, GLint v2, GLint v3);
    // float
    void setUniform1f(GLint al, GLfloat v0);
    void setUniform2f(GLint al, GLfloat v0, GLfloat v1);
    void setUniform3f(GLint al, GLfloat v0, GLfloat v1, GLfloat v2);
    void setUniform4f(GLint al, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3);
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
    //
    // dviser
    void setVertexAttribPointerF32(GLint al, GLint size, GLboolean normalize, GLsizei stride, const GLvoid *v);
    void setVertexAttribPointerI32(GLint al, GLint size, GLsizei stride, const GLvoid *v);
    void enableVertexAttribArray(GLuint  index);
    void disableVertexAttribArray(GLuint  index);
    void setVertexAttribDivisor(GLint al, GLint v);
};


} // namespace graphics
} // namespace ist
#endif // __ist_Graphics_ShaderObject__
