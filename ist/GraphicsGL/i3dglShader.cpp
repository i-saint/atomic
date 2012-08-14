#include "stdafx.h"
#ifdef __ist_with_OpenGL__
#include "ist/Base.h"
#include "i3dglShader.h"

namespace ist {
namespace i3dgl {

template<size_t ShaderType>
ShaderObject<ShaderType>::ShaderObject(Device *dev)
    : super(dev)
{
    m_handle = glCreateShader(ShaderType);
}

template<size_t ShaderType>
ShaderObject<ShaderType>::~ShaderObject()
{
    if(m_handle!=0) {
        glDeleteShader(m_handle);
        m_handle = 0;
    }
}

template<size_t ShaderType>
bool ShaderObject<ShaderType>::compile( const char *src, int length )
{
    glShaderSource(m_handle, 1, &src, &length);
    glCompileShader(m_handle);

    // get errors
    GLint result;
    glGetShaderiv(m_handle, GL_COMPILE_STATUS, &result);
    if(result==GL_FALSE) {
        int length;
        glGetShaderiv(m_handle, GL_INFO_LOG_LENGTH, &length);
        if(length > 0) {
            int l;
            GLchar *info_log = new GLchar[length];
            glGetShaderInfoLog(m_handle, length, &l, info_log);
            istPrint(info_log);
            istAssert(false, "compile failed.");
            delete[] info_log;
        }
        return false;
    }

    return true;
}

template ShaderObject<GL_VERTEX_SHADER>;
template ShaderObject<GL_FRAGMENT_SHADER>;
template ShaderObject<GL_GEOMETRY_SHADER>;

VertexShader::VertexShader( Device *dev, const VertexShaderDesc &desc )
    : super(dev)
{
    compile(desc.source, desc.source_len);
}

VertexShader::~VertexShader()
{
}

PixelShader::PixelShader( Device *dev, const PixelShaderDesc &desc )
    : super(dev)
{
    compile(desc.source, desc.source_len);
}

PixelShader::~PixelShader()
{
}

GeometryShader::GeometryShader( Device *dev, const GeometryShaderDesc &desc )
    : super(dev)
{
    compile(desc.source, desc.source_len);
}

GeometryShader::~GeometryShader()
{
}



ShaderProgram::ShaderProgram(Device *dev, const ShaderProgramDesc &desc)
    : super(dev)
{
    m_handle = glCreateProgram();


    if(desc.vsh) { glAttachShader(m_handle, desc.vsh->getHandle()); }
    if(desc.psh) { glAttachShader(m_handle, desc.psh->getHandle()); }
    if(desc.gsh) { glAttachShader(m_handle, desc.gsh->getHandle()); }
    glLinkProgram(m_handle);

    // get errors
    GLint result;
    glGetProgramiv(m_handle, GL_LINK_STATUS, &result);
    if(result==GL_FALSE) {
        int length;
        glGetProgramiv(m_handle, GL_INFO_LOG_LENGTH, &length);
        if(length > 0) {
            int l;
            GLchar *info_log = new GLchar[length];
            glGetProgramInfoLog(m_handle, length, &l, info_log);
            istPrint(info_log);
            istAssert(false, "link failed.");
            delete[] info_log;
        }
    }
}

ShaderProgram::~ShaderProgram()
{
    if(m_handle!=0) {
        glDeleteProgram(m_handle);
        m_handle = 0;
    }
}


void ShaderProgram::bind()
{
    glUseProgram(m_handle);
}

void ShaderProgram::unbind()
{
    glUseProgram(0);
}


GLint ShaderProgram::getUniformLocation(const char *name) const
{
    GLint ul = glGetUniformLocation(m_handle, name);
    if(ul == -1) {
        istPrint("no such uniform named %s\n", name);
    }
    return ul;
}

GLint ShaderProgram::getAttribLocation(const char *name) const
{
    GLint al = glGetAttribLocation(m_handle, name);
    if(al == -1) {
        istPrint("no such attribute named %s\n", name);
    }
    return al;
}

GLint ShaderProgram::getUniformBlockIndex(const char *name) const
{
    GLint ul = glGetUniformBlockIndex(m_handle, name);
    if(ul == -1) {
        istPrint("no such uniform block named %s\n", name);
    }
    else {
        //GLint block_size = 0;
        //glGetActiveUniformBlockiv(m_handle, ul, GL_UNIFORM_BLOCK_DATA_SIZE, &block_size);
        //istPrint("size of %s: %d\n", name, block_size);
    }
    return ul;
}

void ShaderProgram::setUniformBlock(GLuint uniformBlockIndex, GLuint uniformBindingIndex, GLuint uniformBufferHandle)
{
    glBindBufferBase(GL_UNIFORM_BUFFER, uniformBindingIndex, uniformBufferHandle);
    glUniformBlockBinding(m_handle, uniformBlockIndex, uniformBindingIndex);
}

// uniform variable
// int
void ShaderProgram::setUniform1i(GLint al, GLint v) { glUniform1i(al, v); }
void ShaderProgram::setUniform2i(GLint al, const ivec2& v) { glUniform2i(al, v.x, v.y); }
void ShaderProgram::setUniform3i(GLint al, const ivec3& v) { glUniform3i(al, v.x, v.y, v.z); }
void ShaderProgram::setUniform4i(GLint al, const ivec4& v) { glUniform4i(al, v.x, v.y, v.z, v.w); }

// float
void ShaderProgram::setUniform1f(GLint al, GLfloat v0) { glUniform1f(al, v0); }
void ShaderProgram::setUniform2f(GLint al, const vec2& v) { glUniform2f(al, v.x, v.y); }
void ShaderProgram::setUniform3f(GLint al, const vec3& v) { glUniform3f(al, v.x, v.y, v.z); }
void ShaderProgram::setUniform4f(GLint al, const vec4& v) { glUniform4f(al, v.x, v.y, v.z, v.w); }

// int array
void ShaderProgram::setUniform1iv(GLint al, GLuint count, const GLint *v) { glUniform1iv(al, count, v); }
void ShaderProgram::setUniform2iv(GLint al, GLuint count, const GLint *v) { glUniform2iv(al, count, v); }
void ShaderProgram::setUniform3iv(GLint al, GLuint count, const GLint *v) { glUniform3iv(al, count, v); }
void ShaderProgram::setUniform4iv(GLint al, GLuint count, const GLint *v) { glUniform4iv(al, count, v); }

// float array
void ShaderProgram::setUniform1fv(GLint al, GLuint count, const GLfloat *v) { glUniform1fv(al, count, v); }
void ShaderProgram::setUniform2fv(GLint al, GLuint count, const GLfloat *v) { glUniform2fv(al, count, v); }
void ShaderProgram::setUniform3fv(GLint al, GLuint count, const GLfloat *v) { glUniform3fv(al, count, v); }
void ShaderProgram::setUniform4fv(GLint al, GLuint count, const GLfloat *v) { glUniform4fv(al, count, v); }

// matrix
void ShaderProgram::setUniformMatrix2fv(GLint al, GLuint count, GLboolean transpose, const GLfloat *v) { glUniformMatrix2fv(al, count, transpose, v); }
void ShaderProgram::setUniformMatrix3fv(GLint al, GLuint count, GLboolean transpose, const GLfloat *v) { glUniformMatrix3fv(al, count, transpose, v); }
void ShaderProgram::setUniformMatrix4fv(GLint al, GLuint count, GLboolean transpose, const GLfloat *v) { glUniformMatrix4fv(al, count, transpose, v); }

// attribute variable
// float
void ShaderProgram::setVertexAttrib1f(GLint al, GLfloat v0) { glVertexAttrib1f(al, v0); }
void ShaderProgram::setVertexAttrib2f(GLint al, GLfloat v0, GLfloat v1) { glVertexAttrib2f(al, v0, v1); }
void ShaderProgram::setVertexAttrib3f(GLint al, GLfloat v0, GLfloat v1, GLfloat v2) { glVertexAttrib3f(al, v0, v1, v2); }
void ShaderProgram::setVertexAttrib4f(GLint al, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3) { glVertexAttrib4f(al, v0, v1, v2, v3); }

// float array
void ShaderProgram::setVertexAttrib1fv(GLint al, const GLfloat *v) { glVertexAttrib1fv(al, v); }
void ShaderProgram::setVertexAttrib2fv(GLint al, const GLfloat *v) { glVertexAttrib2fv(al, v); }
void ShaderProgram::setVertexAttrib3fv(GLint al, const GLfloat *v) { glVertexAttrib3fv(al, v); }
void ShaderProgram::setVertexAttrib4fv(GLint al, const GLfloat *v) { glVertexAttrib4fv(al, v); }

GLuint ShaderProgram::getSubroutineIndexVS(const char *name) { return glGetSubroutineIndex(m_handle, GL_VERTEX_SHADER, name); }
GLuint ShaderProgram::getSubroutineIndexGS(const char *name) { return glGetSubroutineIndex(m_handle, GL_GEOMETRY_SHADER, name); }
GLuint ShaderProgram::getSubroutineIndexPS(const char *name) { return glGetSubroutineIndex(m_handle, GL_FRAGMENT_SHADER, name); }
void ShaderProgram::setSubroutineVS(GLsizei count, GLuint *indices) { glUniformSubroutinesuiv(GL_VERTEX_SHADER, count, indices); }
void ShaderProgram::setSubroutineGS(GLsizei count, GLuint *indices) { glUniformSubroutinesuiv(GL_GEOMETRY_SHADER, count, indices); }
void ShaderProgram::setSubroutinePS(GLsizei count, GLuint *indices) { glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, count, indices); }


} // namespace i3d
} // namespace ist
#endif // __ist_with_OpenGL__
