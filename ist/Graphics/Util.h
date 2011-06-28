#ifndef __ist_Graphics_Util__
#define __ist_Graphics_Util__

#include "ShaderObject.h"
#include "FrameBufferObject.h"

namespace ist {
namespace graphics {

bool CreateTexture2DFromFile(Texture2D& tex, const char *filename);
bool CreateTexture2DFromStream(Texture2D& tex, std::istream& st);

bool CreateVertexShaderFromFile(VertexShader& sh, const char *filename);
bool CreateGeometryShaderFromFile(GeometryShader& sh, const char *filename);
bool CreateFragmentShaderFromFile(FragmentShader& sh, const char *filename);
bool CreateVertexShaderFromStream(VertexShader& sh, std::istream& st);
bool CreateGeometryShaderFromStream(GeometryShader& sh, std::istream& st);
bool CreateFragmentShaderFromStream(FragmentShader& sh, std::istream& st);





/// カラーテクスチャだけをbindしたFBO 
/// ポストエフェクトなどに 
template<size_t NumColorBuffers>
class ColorNBuffer : public FrameBufferObject
{
typedef FrameBufferObject super;
private:
    Texture2D m_color[NumColorBuffers];

public:
    bool initialize(GLsizei width, GLsizei height);
    Texture2D* getColorBuffer(size_t i=0) { return &m_color[i]; }
};
typedef ColorNBuffer<1> ColorBuffer;
typedef ColorNBuffer<2> Color2Buffer;
typedef ColorNBuffer<3> Color3Buffer;
typedef ColorNBuffer<4> Color4Buffer;
typedef ColorNBuffer<5> Color5Buffer;
typedef ColorNBuffer<6> Color6Buffer;
typedef ColorNBuffer<7> Color7Buffer;
typedef ColorNBuffer<8> Color8Buffer;



/// デプステクスチャだけをbindしたFBO 
/// 影バッファなどに 
class DepthBuffer : public FrameBufferObject
{
typedef FrameBufferObject super;
private:
    Texture2D m_depth;

public:
    bool initialize(GLsizei width, GLsizei height);
    Texture2D* getDepthBuffer() { return &m_depth; }
};

/// カラーテクスチャとデプスレンダーバッファをbindしたFBO 
template<size_t NumColorBuffers>
class ColorNDepthBuffer : public FrameBufferObject
{
typedef FrameBufferObject super;
public:
    enum FORMAT
    {
        FMT_RGB_U8      = Texture2D::FMT_RGB_U8,
        FMT_RGBA_U8     = Texture2D::FMT_RGBA_U8,
        FMT_RGB_F16     = Texture2D::FMT_RGB_F16,
        FMT_RGBA_F16    = Texture2D::FMT_RGBA_F16,
        FMT_RGB_F32     = Texture2D::FMT_RGB_F32,
        FMT_RGBA_F32    = Texture2D::FMT_RGBA_F32,
        FMT_DEPTH_F32   = Texture2D::FMT_DEPTH_F32,
    };

private:
    Texture2D *m_owned[NumColorBuffers+1];

    Texture2D *m_depth;
    Texture2D *m_color[NumColorBuffers];

    GLsizei m_width;
    GLsizei m_height;

public:
    ColorNDepthBuffer();
    ~ColorNDepthBuffer();
    bool initialize(GLsizei width, GLsizei height, FORMAT=FMT_RGBA_U8);

    GLsizei getWidth() const { return m_width; }
    GLsizei getHeight() const { return m_height; }

    GLsizei getColorBufferNum() const { return NumColorBuffers; }
    Texture2D* getDepthBuffer() { return m_depth; }
    Texture2D* getColorBuffer(size_t i) { return m_color[i]; }

    // initialize() の前に以下の関数で差し替えておくことで代用できる。
    // 設定しなかった場合内部的に作られる。デストラクタで破棄するテクスチャは内部的に作られたものだけ。
    void setDepthBuffer(Texture2D* v) { m_depth=v; }
    void setColorBuffer(size_t i, Texture2D* v) { m_color[i]=v; }
};
typedef ColorNDepthBuffer<1> ColorDepthBuffer;
typedef ColorNDepthBuffer<2> Color2DepthBuffer;
typedef ColorNDepthBuffer<3> Color3DepthBuffer;
typedef ColorNDepthBuffer<4> Color4DepthBuffer;
typedef ColorNDepthBuffer<5> Color5DepthBuffer;
typedef ColorNDepthBuffer<6> Color6DepthBuffer;
typedef ColorNDepthBuffer<7> Color7DepthBuffer;
typedef ColorNDepthBuffer<8> Color8DepthBuffer;


} // namespace graphics
} // namespace ist
#endif // __ist_Graphics_Util__
