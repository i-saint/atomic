#ifndef __ist_Graphics_Util__
#define __ist_Graphics_Util__

#include "ShaderObject.h"
#include "FrameBufferObject.h"

class SFMT;

namespace ist {
namespace graphics {

// 画像ファイル/ストリームからテクスチャ生成
bool CreateTexture2DFromFile(Texture2D& tex, const char *filename);
bool CreateTexture2DFromStream(Texture2D& tex, std::istream& st);

// 乱数テクスチャ生成
bool GenerateRandomTexture(Texture2D &tex, GLsizei width, GLsizei height, Texture2D::FORMAT format);
bool GenerateRandomTexture(Texture2D &tex, GLsizei width, GLsizei height, Texture2D::FORMAT format, SFMT& random);

// ファイル/ストリームから各種シェーダ生成
bool CreateVertexShaderFromFile(VertexShader& sh, const char *filename);
bool CreateGeometryShaderFromFile(GeometryShader& sh, const char *filename);
bool CreateFragmentShaderFromFile(FragmentShader& sh, const char *filename);
bool CreateVertexShaderFromStream(VertexShader& sh, std::istream& st);
bool CreateGeometryShaderFromStream(GeometryShader& sh, std::istream& st);
bool CreateFragmentShaderFromStream(FragmentShader& sh, std::istream& st);
bool CreateVertexShaderFromString(VertexShader& sh, const char* source);
bool CreateGeometryShaderFromString(GeometryShader& sh, const char* source);
bool CreateFragmentShaderFromString(FragmentShader& sh, const char* source);


template<class BufferObjectType>
bool MapAndCopy(BufferObjectType& bo, typename BufferObjectType::USAGE usage, void *data, size_t data_size)
{
    if(void *p = bo.map(usage)) {
        ::memcpy(p, data, data_size);
        bo.unmap();
        return true;
    }
    return false;
}



/// カラーテクスチャだけの FBO 
/// ポストエフェクトなどに 
template<size_t NumColorBuffers>
class ColorNBuffer : public FrameBufferObject
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
    Texture2D *m_owned[NumColorBuffers];

    Texture2D *m_color[NumColorBuffers];
    GLsizei m_width;
    GLsizei m_height;

public:
    ColorNBuffer();
    ~ColorNBuffer();
    bool initialize(GLsizei width, GLsizei height, FORMAT=FMT_RGBA_U8);

    GLsizei getWidth() const { return m_width; }
    GLsizei getHeight() const { return m_height; }

    GLsizei getColorBufferNum() const { return NumColorBuffers; }
    Texture2D* getColorBuffer(size_t i) { return m_color[i]; }

    // initialize() の前に以下の関数で差し替えておくことで代用できる。
    // 設定しなかった場合内部的に作られる。デストラクタで破棄するのは内部的に作られたものだけ。
    void setColorBuffer(size_t i, Texture2D* v) { m_color[i]=v; }
};
typedef ColorNBuffer<1> ColorBuffer;
typedef ColorNBuffer<2> Color2Buffer;
typedef ColorNBuffer<3> Color3Buffer;
typedef ColorNBuffer<4> Color4Buffer;
typedef ColorNBuffer<5> Color5Buffer;
typedef ColorNBuffer<6> Color6Buffer;
typedef ColorNBuffer<7> Color7Buffer;
typedef ColorNBuffer<8> Color8Buffer;



/// デプステクスチャだけの FBO 
/// 影バッファなどに 
class DepthBuffer : public FrameBufferObject
{
typedef FrameBufferObject super;
private:
    Texture2D *m_owned;

    Texture2D *m_depth;
    GLsizei m_width;
    GLsizei m_height;

public:
    DepthBuffer();
    ~DepthBuffer();
    bool initialize(GLsizei width, GLsizei height);

    GLsizei getWidth() const { return m_width; }
    GLsizei getHeight() const { return m_height; }

    Texture2D* getDepthBuffer() { return m_depth; }

    // initialize() の前に以下の関数で差し替えておくことで代用できる。
    // 設定しなかった場合内部的に作られる。デストラクタで破棄するのは内部的に作られたものだけ。
    void setDepthBuffer(Texture2D* v) { m_depth=v; }
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
    // 設定しなかった場合内部的に作られる。デストラクタで破棄するのは内部的に作られたものだけ。
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
