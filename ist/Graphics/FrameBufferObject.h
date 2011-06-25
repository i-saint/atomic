#ifndef __ist_Graphics_FrameBufferObject__
#define __ist_Graphics_FrameBufferObject__

#include "GraphicsResource.h"

namespace ist {
namespace graphics {

    class Texture2D : public GraphicsResource
    {
    public:
        enum FORMAT
        {
            FMT_RGBA_I8,
            FMT_DEPTH_F32,
        };
    private:
        GLuint m_handle;

    public:
        Texture2D();
        ~Texture2D();

        bool initialize(GLsizei width, GLsizei height, FORMAT format, void *data=NULL);

        GLuint getHandle() const { return m_handle; }
        void bind() const;
        void unbind() const;
    };



    class RenderBuffer : public GraphicsResource
    {
    public:
        enum FORMAT
        {
            FMT_RGBA_I8,
            FMT_DEPTH_F32,
        };
    private:
        GLuint m_handle;

    public:
        RenderBuffer();
        ~RenderBuffer();

        bool initialize(GLsizei width, GLsizei height, FORMAT format);

        GLuint getHandle() const { return m_handle; }
        void bind() const;
        void unbind() const;
    };


    class FrameBufferObject : public GraphicsResource
    {
    public:
        enum ATTACH
        {
            ATTACH_COLOR0 = GL_COLOR_ATTACHMENT0,
            ATTACH_COLOR1 = GL_COLOR_ATTACHMENT1,
            ATTACH_COLOR2 = GL_COLOR_ATTACHMENT2,
            ATTACH_COLOR3 = GL_COLOR_ATTACHMENT3,
            ATTACH_COLOR4 = GL_COLOR_ATTACHMENT4,
            ATTACH_COLOR5 = GL_COLOR_ATTACHMENT5,
            ATTACH_COLOR6 = GL_COLOR_ATTACHMENT6,
            ATTACH_COLOR7 = GL_COLOR_ATTACHMENT7,
            ATTACH_COLOR8 = GL_COLOR_ATTACHMENT8,
            ATTACH_COLOR9 = GL_COLOR_ATTACHMENT9,
            ATTACH_COLOR10 = GL_COLOR_ATTACHMENT10,
            ATTACH_COLOR11 = GL_COLOR_ATTACHMENT11,
            ATTACH_COLOR12 = GL_COLOR_ATTACHMENT12,
            ATTACH_COLOR13 = GL_COLOR_ATTACHMENT13,
            ATTACH_COLOR14 = GL_COLOR_ATTACHMENT14,
            ATTACH_COLOR15 = GL_COLOR_ATTACHMENT15,
            ATTACH_DEPTH  = GL_DEPTH_ATTACHMENT,
        };
    private:
        GLuint m_handle;

    public:
        FrameBufferObject();
        ~FrameBufferObject();

        bool attachRenderBuffer(RenderBuffer& tex, ATTACH attach);
        bool attachTexture(Texture2D& rb, ATTACH attach, GLint level=0);

        GLuint getHandle() const { return m_handle; }
        void bind() const;
        void unbind() const;
    };


  //class FrameBufferObject
  //{
  //private:
  //  GLuint m_fbo;
  //  std::vector<GLuint> m_texture;
  //  std::vector<GLuint> m_renderbuffer;
  //  GLsizei m_width;
  //  GLsizei m_height;

  //public:
  //  FrameBufferObject(GLsizei width, GLsizei height)
  //    : m_fbo(0), m_width(width), m_height(height), m_prev_target(0)
  //  {
  //    glGenFramebuffersEXT(1, &m_fbo);
  //  }

  //  virtual ~FrameBufferObject()
  //  {
  //    if(!m_texture.empty()) {
  //      glDeleteTextures(m_texture.size(), &m_texture[0]);
  //    }
  //    if(!m_renderbuffer.empty()) {
  //      glDeleteRenderbuffersEXT(m_renderbuffer.size(), &m_renderbuffer[0]);
  //    }
  //    glDeleteFramebuffersEXT(1, &m_fbo);
  //  }

  //  GLuint getFrameBufferObjectHandle() { return m_fbo; }
  //  GLuint getRenderBufferHandle(GLuint i) { return m_renderbuffer[i]; }
  //  GLuint getTextureHandle(GLuint i) { return m_texture[i]; }
  //  GLuint getRenderBufferSize() const { return m_renderbuffer.size(); }
  //  GLuint getTextureSize() const { return m_texture.size(); }
  //  GLsizei getWidth() const { return m_width; }
  //  GLsizei getHeight() const { return m_height; }

  //  /// format: GL_RGBA,GL_DEPTH_COMPONENT 等 
  //  /// type: GL_UNSIGNED_BYTE,GL_UNSIGNED_INT 等 
  //  /// attachment: GL_COLOR_ATTACHMENT0_EXT,GL_DEPTH_ATTACHMENT_EXT 等 
  //  void attachTexture(GLint format, GLint type, GLint attachment)
  //  {
  //    GLuint name = 0;
  //    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  //    glGenTextures(1, &name);
  //    glBindTexture(GL_TEXTURE_2D, name);
  //    glTexImage2D(GL_TEXTURE_2D, 0, format, getWidth(), getHeight(), 0, format, type, 0);
  //    CheckGLError();

  //    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, m_fbo);
  //    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, attachment, GL_TEXTURE_2D, name, 0);
  //    CheckGLError();
  //    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
  //    glBindTexture(GL_TEXTURE_2D, 0);

  //    m_texture.push_back(name);
  //  }

  //  /// format: GL_RGBA,GL_DEPTH_COMPONENT 等 
  //  /// attachment: GL_COLOR_ATTACHMENT0_EXT,GL_DEPTH_ATTACHMENT_EXT 等 
  //  void attachRenderBuffer(GLint format, GLint attachment)
  //  {
  //    GLuint name = 0;
  //    glGenRenderbuffersEXT(1, &name);
  //    glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, name);
  //    glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, format, getWidth(), getHeight());

  //    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, m_fbo);
  //    glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, attachment, GL_RENDERBUFFER_EXT, name);
  //    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
  //    glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, 0);

  //    m_renderbuffer.push_back(name);
  //  }

  //  /// レンダターゲットを切り替え、ビューポートをFBOのサイズに合わせる 
  //  void enable()
  //  {
  //    glGetIntegerv(GL_FRAMEBUFFER_BINDING_EXT, &m_prev_target);
  //    glGetIntegerv(GL_VIEWPORT, m_viewport);
  //    glViewport(0,0, getWidth(), getHeight());

  //    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, m_fbo);
  //    CheckGLError();
  //  }

  //  /// レンダターゲットとビューポートを元のサイズに戻す 
  //  void disable()
  //  {
  //    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, m_prev_target);
  //    CheckGLError();

  //    glViewport(m_viewport[0],m_viewport[1], m_viewport[2],m_viewport[3]);
  //  }

  //  /// テクスチャをbindする 
  //  void assign(GLuint i=0)
  //  {
  //    if(i<m_texture.size()) {
  //      glBindTexture(GL_TEXTURE_2D, m_texture[i]);
  //      CheckGLError();
  //    }
  //  }

  //  /// テクスチャを無効化 (0にbind) 
  //  void disassign()
  //  {
  //    glBindTexture(GL_TEXTURE_2D, 0);
  //  }
  //};








} // graphics
} // ist
#endif // __ist_Graphics_FrameBufferObject__
