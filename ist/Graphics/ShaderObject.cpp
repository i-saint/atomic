#include "stdafx.h"
#include "../Base/Assert.h"
#include "GraphicsAssert.h"
#include "ShaderObject.h"

namespace ist {
namespace graphics {

template<size_t ShaderType>
ShaderObject<ShaderType>::ShaderObject()
  : m_handle(0)
{
    // create
    m_handle = glCreateShader(SHADER_TYPE);
    CheckGLError();
}

template<size_t ShaderType>
ShaderObject<ShaderType>::~ShaderObject()
{
    glDeleteShader(m_handle);
}

template<size_t ShaderType>
bool ShaderObject<ShaderType>::initialize(const char *src, int length)
{
    // set shader source
    glShaderSource(m_handle, 1, &src, &length);
    if(glGetError() != GL_NO_ERROR) {
        return false;
    }
    // compile
    glCompileShader(m_handle);

    // get errors
    GLint result;
    glGetShaderiv(m_handle, GL_COMPILE_STATUS, &result);
    if(glGetError()!=GL_NO_ERROR || result==GL_FALSE) {
        int length;
        glGetShaderiv(m_handle, GL_INFO_LOG_LENGTH, &length);
        if(length > 0) {
            int l;
            GLchar *info_log = new GLchar[length];
            glGetShaderInfoLog(m_handle, length, &l, info_log);
            IST_ASSERT(info_log);
            delete[] info_log;
        }
        return false;
    }

    return true;
}

template<size_t ShaderType>
void ShaderObject<ShaderType>::finalize()
{
}

template ShaderObject<GL_VERTEX_SHADER>;
template ShaderObject<GL_FRAGMENT_SHADER>;
template ShaderObject<GL_GEOMETRY_SHADER>;



ProgramObject::ProgramObject()
: m_handle(0)
{
    m_handle = glCreateProgram();
    CheckGLError();
}

ProgramObject::~ProgramObject()
{
    glDeleteProgram(m_handle);
}

void ProgramObject::attachVertexShader(VertexShader *sh)
{
    if(sh) {
        glAttachShader(m_handle, sh->getHandle());
        CheckGLError();
    }
}

void ProgramObject::attachGeometryShader(GeometryShader *sh)
{
    if(sh) {
        glAttachShader(m_handle, sh->getHandle());
        CheckGLError();
    }
}

void ProgramObject::attachFragmentShader(FragmentShader *sh)
{
    if(sh) {
        glAttachShader(m_handle, sh->getHandle());
        CheckGLError();
    }
}

bool ProgramObject::link()
{
    // link
    glLinkProgram(m_handle);
    CheckGLError();

    // get errors
    GLint result;
    glGetProgramiv(m_handle, GL_LINK_STATUS, &result);
    if(glGetError() != GL_NO_ERROR || result==GL_FALSE) {
        int length;
        glGetProgramiv(m_handle, GL_INFO_LOG_LENGTH, &length);
        if(length > 0) {
            int l;
            GLchar *info_log = new GLchar[length];
            glGetProgramInfoLog(m_handle, length, &l, info_log);
            IST_ASSERT(info_log);
            delete[] info_log;
        }
        return false;
    }

    return true;
}

bool ProgramObject::initialize(VertexShader *vsh, GeometryShader *gsh, FragmentShader *fsh)
{
    attachVertexShader(vsh);
    attachGeometryShader(gsh);
    attachFragmentShader(fsh);
    return link();
}

void ProgramObject::finalize()
{
}


void ProgramObject::bind() const
{
    glUseProgram(m_handle);
    CheckGLError();
}

void ProgramObject::unbind() const
{
    glUseProgram(0);
}


GLint ProgramObject::getUniformLocation(const char *name)
{
    GLint ul = glGetUniformLocation(m_handle, name);
    if(ul == -1) {
        IST_ASSERT("no such uniform named %s", name);
    }
    return ul;
}

GLint ProgramObject::getAttribLocation(const char *name)
{
    GLint al = glGetAttribLocation(m_handle, name);
    if(al == -1) {
        IST_ASSERT("no such attribute named %s", name);
    }
    return al;
}

GLint ProgramObject::getUniformBlockIndex(const char *name)
{
    GLint ul = glGetUniformBlockIndex(m_handle, name);
    if(ul == -1) {
        IST_ASSERT("no such uniform block named %s", name);
    }
    return ul;
}

void ProgramObject::setUniformBlockBinding(GLuint uniformBlockIndex, GLuint uniformBufferHandle)
{
    glUniformBlockBinding(m_handle, uniformBlockIndex, uniformBufferHandle);
}

// uniform variable
// int
void ProgramObject::setUniform1i(GLint al, GLint v0) { glUniform1i(al, v0); }
void ProgramObject::setUniform2i(GLint al, GLint v0, GLint v1) { glUniform2i(al, v0, v1); }
void ProgramObject::setUniform3i(GLint al, GLint v0, GLint v1, GLint v2) { glUniform3i(al, v0, v1, v2); }
void ProgramObject::setUniform4i(GLint al, GLint v0, GLint v1, GLint v2, GLint v3) { glUniform4i(al, v0, v1, v2, v3); }

// float
void ProgramObject::setUniform1f(GLint al, GLfloat v0) { glUniform1f(al, v0); }
void ProgramObject::setUniform2f(GLint al, GLfloat v0, GLfloat v1) { glUniform2f(al, v0, v1); }
void ProgramObject::setUniform3f(GLint al, GLfloat v0, GLfloat v1, GLfloat v2) { glUniform3f(al, v0, v1, v2); }
void ProgramObject::setUniform4f(GLint al, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3) { glUniform4f(al, v0, v1, v2, v3); }

// int array
void ProgramObject::setUniform1iv(GLint al, GLuint count, const GLint *v) { glUniform1iv(al, count, v); }
void ProgramObject::setUniform2iv(GLint al, GLuint count, const GLint *v) { glUniform2iv(al, count, v); }
void ProgramObject::setUniform3iv(GLint al, GLuint count, const GLint *v) { glUniform3iv(al, count, v); }
void ProgramObject::setUniform4iv(GLint al, GLuint count, const GLint *v) { glUniform4iv(al, count, v); }

// float array
void ProgramObject::setUniform1fv(GLint al, GLuint count, const GLfloat *v) { glUniform1fv(al, count, v); }
void ProgramObject::setUniform2fv(GLint al, GLuint count, const GLfloat *v) { glUniform2fv(al, count, v); }
void ProgramObject::setUniform3fv(GLint al, GLuint count, const GLfloat *v) { glUniform3fv(al, count, v); }
void ProgramObject::setUniform4fv(GLint al, GLuint count, const GLfloat *v) { glUniform4fv(al, count, v); }

// matrix
void ProgramObject::setUniformMatrix2fv(GLint al, GLuint count, GLboolean transpose, const GLfloat *v) { glUniformMatrix2fv(al, count, transpose, v); }
void ProgramObject::setUniformMatrix3fv(GLint al, GLuint count, GLboolean transpose, const GLfloat *v) { glUniformMatrix3fv(al, count, transpose, v); }
void ProgramObject::setUniformMatrix4fv(GLint al, GLuint count, GLboolean transpose, const GLfloat *v) { glUniformMatrix4fv(al, count, transpose, v); }

// attribute variable
// float
void ProgramObject::setVertexAttrib1f(GLint al, GLfloat v0) { glVertexAttrib1f(al, v0); }
void ProgramObject::setVertexAttrib2f(GLint al, GLfloat v0, GLfloat v1) { glVertexAttrib2f(al, v0, v1); }
void ProgramObject::setVertexAttrib3f(GLint al, GLfloat v0, GLfloat v1, GLfloat v2) { glVertexAttrib3f(al, v0, v1, v2); }
void ProgramObject::setVertexAttrib4f(GLint al, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3) { glVertexAttrib4f(al, v0, v1, v2, v3); }

// float array
void ProgramObject::setVertexAttrib1fv(GLint al, const GLfloat *v) { glVertexAttrib1fv(al, v); }
void ProgramObject::setVertexAttrib2fv(GLint al, const GLfloat *v) { glVertexAttrib2fv(al, v); }
void ProgramObject::setVertexAttrib3fv(GLint al, const GLfloat *v) { glVertexAttrib3fv(al, v); }
void ProgramObject::setVertexAttrib4fv(GLint al, const GLfloat *v) { glVertexAttrib4fv(al, v); }

void ProgramObject::setVertexAttribPointerF32(GLint al, GLint size, GLboolean normalize, GLsizei stride, const GLvoid *v) { glVertexAttribPointer(al, size, GL_FLOAT, normalize, stride, v); }
void ProgramObject::setVertexAttribPointerI32(GLint al, GLint size, GLsizei stride, const GLvoid *v) { glVertexAttribIPointer(al, size, GL_INT, stride, v); }

void ProgramObject::enableVertexAttribArray(GLuint i) { glEnableVertexAttribArray(i); }
void ProgramObject::disableVertexAttribArray(GLuint i) { glDisableVertexAttribArray(i); }
void ProgramObject::setVertexAttribDivisor(GLint al, GLint v) { glVertexAttribDivisor(al, v); }


} // namespace graphics
} // namespace ist
