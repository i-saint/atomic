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
private:
    Texture2D m_color[NumColorBuffers];
    RenderBuffer m_depth;

public:
    bool initialize(GLsizei width, GLsizei height);
    Texture2D* getColorBuffer(size_t i) { return &m_color[i]; }
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
