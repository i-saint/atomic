#ifndef ist_i3dgl_Shader_h
#define ist_i3dgl_Shader_h

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

    bool  compile(const char *src, int length);
public:
};

class VertexShader : public ShaderObject<GL_VERTEX_SHADER>
{
I3DGL_DECLARE_DEVICE_RESOURCE(VertexShader);
typedef ShaderObject<GL_VERTEX_SHADER> super;
private:
    VertexShader(Device *dev, const VertexShaderDesc &desc);
    ~VertexShader();

public:
};

class PixelShader : public ShaderObject<GL_FRAGMENT_SHADER>
{
I3DGL_DECLARE_DEVICE_RESOURCE(PixelShader);
typedef ShaderObject<GL_FRAGMENT_SHADER> super;
private:
    PixelShader(Device *dev, const PixelShaderDesc &desc);
    ~PixelShader();

public:
};

class GeometryShader : public ShaderObject<GL_GEOMETRY_SHADER>
{
I3DGL_DECLARE_DEVICE_RESOURCE(GeometryShader);
typedef ShaderObject<GL_GEOMETRY_SHADER> super;
private:
    GeometryShader(Device *dev, const GeometryShaderDesc &desc);
    ~GeometryShader();

public:
};


class ShaderProgram : public DeviceResource
{
I3DGL_DECLARE_DEVICE_RESOURCE(ShaderProgram);
typedef DeviceResource super;
public:
    GLint getUniformLocation(const char *name) const;
    GLint getAttribLocation(const char *name) const;

    GLint getUniformBlockIndex(const char *name) const;
    void setUniformBlock(GLuint uniformBlockIndex, GLuint uniformBindingIndex, GLuint uniformBufferHandle);

    void setSampler(GLint al, GLint v);

private:
    void bind();
    void unbind();
    ShaderProgram(Device *dev, const ShaderProgramDesc &desc);
    ~ShaderProgram();
};


} // namespace i3d
} // namespace ist
#endif // ist_i3dgl_Shader_h
