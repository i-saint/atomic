#include "istPCH.h"
#include "ist/Base.h"
#ifdef ist_with_OpenGL
#include "i3dglTexture.h"

namespace ist {
namespace i3dgl {

bool DetectGLFormat(I3D_COLOR_FORMAT fmt, GLint &internal_format, GLint &format, GLint &type);




Sampler::Sampler( Device *dev, const SamplerDesc &desc )
    : super(dev)
    , m_desc(desc)
{
    glGenSamplers(1, &m_handle);
    glSamplerParameteri(getHandle(), GL_TEXTURE_WRAP_S, m_desc.wrap_s);
    glSamplerParameteri(getHandle(), GL_TEXTURE_WRAP_T, m_desc.wrap_t);
    glSamplerParameteri(getHandle(), GL_TEXTURE_WRAP_R, m_desc.wrap_r);
    glSamplerParameteri(getHandle(), GL_TEXTURE_MIN_FILTER, m_desc.filter_min);
    glSamplerParameteri(getHandle(), GL_TEXTURE_MAG_FILTER, m_desc.filter_mag);
}

Sampler::~Sampler()
{
    glDeleteSamplers(1, &m_handle);
}

void Sampler::bind( uint32 slot )
{
    glBindSampler(slot, getHandle());
}

void Sampler::unbind( uint32 slot )
{
    glBindSampler(slot, 0);
}



Texture::Texture( Device *dev ) : super(dev)
{
}


const int Texture1D::TEXTURE_TYPE = GL_TEXTURE_1D;

Texture1D::Texture1D(Device *dev, const Texture1DDesc &desc)
    : super(dev)
    , m_desc(desc)
{
    glGenTextures(1, &m_handle);

    GLint internal_format = 0;
    GLint format = 0;
    GLint type = 0;
    DetectGLFormat(m_desc.format, internal_format, format, type);

    bind();
    if(m_desc.format<I3D_RGB_DXT1) {
        glTexImage1D( TEXTURE_TYPE, 0, internal_format, m_desc.size, 0, format, type, m_desc.data );
    }
    else {
        glCompressedTexImage1D( TEXTURE_TYPE, 0, internal_format, m_desc.size, 0, m_desc.data_size, m_desc.data );
    }
    if(desc.mipmap != 0) {
        glGenerateMipmap(TEXTURE_TYPE);
    }
    unbind();
}

Texture1D::~Texture1D()
{
    if(m_handle!=0) {
        glDeleteTextures(1, &m_handle);
        m_handle = 0;
    }
}

void Texture1D::update(uint32 mip_level, uint32 pos, uint32 size, void *data)
{
    GLint internal_format = 0;
    GLint format = 0;
    GLint type = 0;
    DetectGLFormat(m_desc.format, internal_format, format, type);

    bind();
    glTexSubImage1D(TEXTURE_TYPE, mip_level, pos, size, format, type, data);
    unbind();
}

void Texture1D::generateMips()
{
    bind();
    glGenerateMipmap(TEXTURE_TYPE);
    unbind();
}

void Texture1D::bind() const
{
    glBindTexture(TEXTURE_TYPE, m_handle);
}
void Texture1D::unbind() const
{
    glBindTexture(TEXTURE_TYPE, 0);
}

void Texture1D::bind(uint32 slot) const
{
    glActiveTexture(GL_TEXTURE0+slot);
    bind();
}
void Texture1D::unbind(uint32 slot) const
{
    glActiveTexture(GL_TEXTURE0+slot);
    unbind();
}

const Texture1DDesc& Texture1D::getDesc() const { return m_desc; }




const int Texture2D::TEXTURE_TYPE = GL_TEXTURE_2D;

Texture2D::Texture2D(Device *dev, const Texture2DDesc &desc)
    : super(dev)
    , m_desc(desc)
{
    glGenTextures(1, &m_handle);

    GLint internal_format = 0;
    GLint format = 0;
    GLint type = 0;
    DetectGLFormat(m_desc.format, internal_format, format, type);

    bind();
    if(m_desc.format<I3D_RGB_DXT1) {
        glTexImage2D( TEXTURE_TYPE, 0, internal_format, m_desc.size.x, m_desc.size.y, 0, format, type, m_desc.data );
    }
    else {
        glCompressedTexImage2D( TEXTURE_TYPE, 0, internal_format, m_desc.size.x, m_desc.size.y, 0, m_desc.data_size, m_desc.data );
    }
    if(desc.mipmap != 0) {
        glGenerateMipmap(TEXTURE_TYPE);
    }
    unbind();
}

Texture2D::~Texture2D()
{
    if(m_handle!=0) {
        glDeleteTextures(1, &m_handle);
        m_handle = 0;
    }
}

void Texture2D::update(uint32 mip_level, const uvec2 &pos, const uvec2 &size, void *data)
{
    if(size.x-pos.x > m_desc.size.x || size.y-pos.y > m_desc.size.y) {
        istAssert(false, "exceeded texture size.\n");
    }
    GLint internal_format = 0;
    GLint format = 0;
    GLint type = 0;
    DetectGLFormat(m_desc.format, internal_format, format, type);

    bind();
    glTexSubImage2D(TEXTURE_TYPE, mip_level, pos.x, pos.y, size.x, size.y, format, type, data);
    unbind();
}

void Texture2D::generateMips()
{
    bind();
    glGenerateMipmap(TEXTURE_TYPE);
    unbind();
}

void Texture2D::bind() const
{
    glBindTexture(TEXTURE_TYPE, m_handle);
}
void Texture2D::unbind() const
{
    glBindTexture(TEXTURE_TYPE, 0);
}

void Texture2D::bind(uint32 slot) const
{
    glActiveTexture(GL_TEXTURE0+slot);
    bind();
}
void Texture2D::unbind(uint32 slot) const
{
    glActiveTexture(GL_TEXTURE0+slot);
    unbind();
}

const Texture2DDesc& Texture2D::getDesc() const { return m_desc; }




const int Texture3D::TEXTURE_TYPE = GL_TEXTURE_3D;

Texture3D::Texture3D(Device *dev, const Texture3DDesc &desc)
    : super(dev)
    , m_desc(desc)
{
    glGenTextures(1, &m_handle);

    GLint internal_format = 0;
    GLint format = 0;
    GLint type = 0;
    DetectGLFormat(m_desc.format, internal_format, format, type);

    bind();
    if(m_desc.format<I3D_RGB_DXT1) {
        glTexImage3D( TEXTURE_TYPE, 0, internal_format, m_desc.size.x, m_desc.size.y, m_desc.size.z, 0, format, type, m_desc.data );
    }
    else {
        glCompressedTexImage3D( TEXTURE_TYPE, 0, internal_format, m_desc.size.x, m_desc.size.y, m_desc.size.z, 0, m_desc.data_size, m_desc.data );
    }
    if(desc.mipmap != 0) {
        glGenerateMipmap(TEXTURE_TYPE);
    }
    unbind();
}

Texture3D::~Texture3D()
{
    if(m_handle!=0) {
        glDeleteTextures(1, &m_handle);
        m_handle = 0;
    }
}

void Texture3D::update(uint32 mip_level, const uvec3 &pos, const uvec3 &size, void *data)
{
    if(size.x-pos.x > m_desc.size.x || size.y-pos.y > m_desc.size.y || size.z-pos.z > m_desc.size.z) {
        istAssert(false, "exceeded texture size.\n");
    }
    GLint internal_format = 0;
    GLint format = 0;
    GLint type = 0;
    DetectGLFormat(m_desc.format, internal_format, format, type);

    bind();
    glTexSubImage3D(TEXTURE_TYPE, mip_level, pos.x, pos.y, pos.z, size.x, size.y, size.z, format, type, data);
    unbind();
}

void Texture3D::generateMips()
{
    bind();
    glGenerateMipmap(TEXTURE_TYPE);
    unbind();
}

void Texture3D::bind() const
{
    glBindTexture(TEXTURE_TYPE, m_handle);
}
void Texture3D::unbind() const
{
    glBindTexture(TEXTURE_TYPE, 0);
}

void Texture3D::bind(uint32 slot) const
{
    glActiveTexture(GL_TEXTURE0+slot);
    bind();
}
void Texture3D::unbind(uint32 slot) const
{
    glActiveTexture(GL_TEXTURE0+slot);
    unbind();
}

const Texture3DDesc& Texture3D::getDesc() const { return m_desc; }

} // namespace i3dgl
} // namespace ist
#endif // ist_with_OpenGL
