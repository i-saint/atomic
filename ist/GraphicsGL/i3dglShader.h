#ifndef __ist_i3dgl_Shader__
#define __ist_i3dgl_Shader__

#include "i3dglDeviceResource.h"

namespace ist {
namespace i3dgl {

template<size_t ShaderType>
class ShaderObject : public DeviceResource
{
I3DGL_DECLARE_DEVICE_RESOURCE(ShaderObject);
typedef DeviceResource super;
protected:
    ShaderObject(Device *dev);
    ~ShaderObject();

public:
    bool  compile(const char *src, int length);
};

class VertexShader : public ShaderObject<GL_VERTEX_SHADER>
{
I3DGL_DECLARE_DEVICE_RESOURCE(VertexShader);
typedef ShaderObject<GL_VERTEX_SHADER> super;
private:
    VertexShader(Device *dev) : super(dev) {}
    ~VertexShader() {}

public:
};

class PixelShader : public ShaderObject<GL_FRAGMENT_SHADER>
{
I3DGL_DECLARE_DEVICE_RESOURCE(PixelShader);
typedef ShaderObject<GL_FRAGMENT_SHADER> super;
private:
    PixelShader(Device *dev) : super(dev) {}
    ~PixelShader() {}

public:
};

class GeometryShader : public ShaderObject<GL_GEOMETRY_SHADER>
{
I3DGL_DECLARE_DEVICE_RESOURCE(GeometryShader);
typedef ShaderObject<GL_GEOMETRY_SHADER> super;
private:
    GeometryShader(Device *dev) : super(dev) {}
    ~GeometryShader() {}

public:
};


class ShaderProgram : public DeviceResource
{
I3DGL_DECLARE_DEVICE_RESOURCE(ShaderProgram);
typedef DeviceResource super;
private:
    ShaderProgram(Device *dev);
    ~ShaderProgram();

public:
    bool link(VertexShader *vsh, PixelShader *fsh, GeometryShader *gsh=NULL);
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


} // namespace i3d
} // namespace ist
#endif // __ist_i3dgl_Shader__
