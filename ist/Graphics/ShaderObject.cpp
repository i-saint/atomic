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
    m_handle = glCreateShaderObjectARB(SHADER_TYPE);
    CheckGLError();
}

template<size_t ShaderType>
ShaderObject<ShaderType>::~ShaderObject()
{
    glDeleteObjectARB(m_handle);
}

template<size_t ShaderType>
bool ShaderObject<ShaderType>::initialize(const char *src, int length)
{
    // set shader source
    glShaderSourceARB(m_handle, 1, &src, &length);
    if(glGetError() != GL_NO_ERROR) {
        return false;
    }
    // compile
    glCompileShader(m_handle);

    // get errors
    GLint result;
    glGetObjectParameterivARB(m_handle, GL_OBJECT_COMPILE_STATUS_ARB, &result);
    if(glGetError()!=GL_NO_ERROR || result==GL_FALSE) {
        int length;
        glGetObjectParameterivARB(m_handle, GL_OBJECT_INFO_LOG_LENGTH_ARB, &length);
        if(length > 0) {
            int l;
            GLcharARB *info_log = new GLcharARB[length];
            glGetInfoLogARB(m_handle, length, &l, info_log);
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

template ShaderObject<GL_VERTEX_SHADER_ARB>;
template ShaderObject<GL_FRAGMENT_SHADER_ARB>;
template ShaderObject<GL_GEOMETRY_SHADER_ARB>;



ProgramObject::ProgramObject()
: m_handle(0)
{
    m_handle = glCreateProgramObjectARB();
    CheckGLError();
}

ProgramObject::~ProgramObject()
{
    glDeleteObjectARB(m_handle);
}

void ProgramObject::attachVertexShader(VertexShader *sh)
{
    if(sh) {
        glAttachObjectARB(m_handle, sh->getHandle());
        CheckGLError();
    }
}

void ProgramObject::attachGeometryShader(GeometryShader *sh)
{
    if(sh) {
        glAttachObjectARB(m_handle, sh->getHandle());
        CheckGLError();
    }
}

void ProgramObject::attachFragmentShader(FragmentShader *sh)
{
    if(sh) {
        glAttachObjectARB(m_handle, sh->getHandle());
        CheckGLError();
    }
}

bool ProgramObject::link()
{
    // link
    glLinkProgramARB(m_handle);

    // get errors
    GLint result;
    glGetObjectParameterivARB(m_handle, GL_OBJECT_LINK_STATUS_ARB, &result);
    if(glGetError() != GL_NO_ERROR || result==GL_FALSE) {
        int length;
        glGetObjectParameterivARB(m_handle, GL_OBJECT_INFO_LOG_LENGTH_ARB, &length);
        if(length > 0) {
            int l;
            GLcharARB *info_log = new GLcharARB[length];
            glGetInfoLogARB(m_handle, length, &l, info_log);
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
    link();
}

void ProgramObject::finalize()
{
}


void ProgramObject::bind() const
{
    glUseProgramObjectARB(m_handle);
    CheckGLError();
}

void ProgramObject::unbind() const
{
    glUseProgramObjectARB(0);
}


GLint ProgramObject::getUniformLocation(const char *name)
{
    GLint ul = glGetUniformLocationARB(m_handle, name);
    if(ul == -1) {
        IST_ASSERT("no such uniform named %s", name);
    }
    return ul;
}

GLint ProgramObject::getAttribLocation(const char *name)
{
    GLint al = glGetAttribLocationARB(m_handle, name);
    if(al == -1) {
        IST_ASSERT("no such attribute named %s", name);
    }
    return al;
}

// uniform variable
// int
void ProgramObject::setUniform1i(GLint al, GLint v0) { glUniform1iARB(al, v0); }
void ProgramObject::setUniform2i(GLint al, GLint v0, GLint v1) { glUniform2iARB(al, v0, v1); }
void ProgramObject::setUniform3i(GLint al, GLint v0, GLint v1, GLint v2) { glUniform3iARB(al, v0, v1, v2); }
void ProgramObject::setUniform4i(GLint al, GLint v0, GLint v1, GLint v2, GLint v3) { glUniform4iARB(al, v0, v1, v2, v3); }

// float
void ProgramObject::setUniform1f(GLint al, GLfloat v0) { glUniform1fARB(al, v0); }
void ProgramObject::setUniform2f(GLint al, GLfloat v0, GLfloat v1) { glUniform2fARB(al, v0, v1); }
void ProgramObject::setUniform3f(GLint al, GLfloat v0, GLfloat v1, GLfloat v2) { glUniform3fARB(al, v0, v1, v2); }
void ProgramObject::setUniform4f(GLint al, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3) { glUniform4fARB(al, v0, v1, v2, v3); }

// int array
void ProgramObject::setUniform1iv(GLint al, GLuint count, const GLint *v) { glUniform1ivARB(al, count, v); }
void ProgramObject::setUniform2iv(GLint al, GLuint count, const GLint *v) { glUniform2ivARB(al, count, v); }
void ProgramObject::setUniform3iv(GLint al, GLuint count, const GLint *v) { glUniform3ivARB(al, count, v); }
void ProgramObject::setUniform4iv(GLint al, GLuint count, const GLint *v) { glUniform4ivARB(al, count, v); }

// float array
void ProgramObject::setUniform1fv(GLint al, GLuint count, const GLfloat *v) { glUniform1fvARB(al, count, v); }
void ProgramObject::setUniform2fv(GLint al, GLuint count, const GLfloat *v) { glUniform2fvARB(al, count, v); }
void ProgramObject::setUniform3fv(GLint al, GLuint count, const GLfloat *v) { glUniform3fvARB(al, count, v); }
void ProgramObject::setUniform4fv(GLint al, GLuint count, const GLfloat *v) { glUniform4fvARB(al, count, v); }

// matrix
void ProgramObject::setUniformMatrix2fv(GLint al, GLuint count, GLboolean transpose, const GLfloat *v) { glUniformMatrix2fvARB(al, count, transpose, v); }
void ProgramObject::setUniformMatrix3fv(GLint al, GLuint count, GLboolean transpose, const GLfloat *v) { glUniformMatrix3fvARB(al, count, transpose, v); }
void ProgramObject::setUniformMatrix4fv(GLint al, GLuint count, GLboolean transpose, const GLfloat *v) { glUniformMatrix4fvARB(al, count, transpose, v); }

// attribute variable
// float
void ProgramObject::setVertexAttrib1f(GLint al, GLfloat v0) { glVertexAttrib1fARB(al, v0); }
void ProgramObject::setVertexAttrib2f(GLint al, GLfloat v0, GLfloat v1) { glVertexAttrib2fARB(al, v0, v1); }
void ProgramObject::setVertexAttrib3f(GLint al, GLfloat v0, GLfloat v1, GLfloat v2) { glVertexAttrib3fARB(al, v0, v1, v2); }
void ProgramObject::setVertexAttrib4f(GLint al, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3) { glVertexAttrib4fARB(al, v0, v1, v2, v3); }

// float array
void ProgramObject::setVertexAttrib1fv(GLint al, const GLfloat *v) { glVertexAttrib1fvARB(al, v); }
void ProgramObject::setVertexAttrib2fv(GLint al, const GLfloat *v) { glVertexAttrib2fvARB(al, v); }
void ProgramObject::setVertexAttrib3fv(GLint al, const GLfloat *v) { glVertexAttrib3fvARB(al, v); }
void ProgramObject::setVertexAttrib4fv(GLint al, const GLfloat *v) { glVertexAttrib4fvARB(al, v); }


} // namespace graphics
} // namespace ist
