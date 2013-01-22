#ifndef __ist_i3dgl_Util__
#define __ist_i3dgl_Util__

#include "i3dglShader.h"
#include "i3dglRenderTarget.h"

class SFMT;

namespace ist {
namespace i3dgl {

// 画像ファイル/ストリームからテクスチャ生成
istInterModule Texture2D* CreateTexture2DFromFile(Device *dev, const char *filename, I3D_COLOR_FORMAT format=I3D_COLOR_UNKNOWN);
istInterModule Texture2D* CreateTexture2DFromStream(Device *dev, IBinaryStream &st, I3D_COLOR_FORMAT format=I3D_COLOR_UNKNOWN);
istInterModule Texture2D* CreateTexture2DFromImage(Device *dev, Image &img, I3D_COLOR_FORMAT format=I3D_COLOR_UNKNOWN);

// 乱数テクスチャ生成
istInterModule Texture2D* GenerateRandomTexture(Device *dev, const uvec2 &size, I3D_COLOR_FORMAT format);
istInterModule Texture2D* GenerateRandomTexture(Device *dev, const uvec2 &size, I3D_COLOR_FORMAT format, SFMT& random);

// ファイル/ストリームから各種シェーダ生成
istInterModule VertexShader*   CreateVertexShaderFromFile(Device *dev, const char *filename);
istInterModule GeometryShader* CreateGeometryShaderFromFile(Device *dev, const char *filename);
istInterModule PixelShader*    CreatePixelShaderFromFile(Device *dev, const char *filename);

istInterModule VertexShader*   CreateVertexShaderFromStream(Device *dev, IBinaryStream &st);
istInterModule GeometryShader* CreateGeometryShaderFromStream(Device *dev, IBinaryStream &st);
istInterModule PixelShader*    CreatePixelShaderFromStream(Device *dev, IBinaryStream &st);

istInterModule VertexShader*   CreateVertexShaderFromString(Device *dev, const stl::string &source);
istInterModule GeometryShader* CreateGeometryShaderFromString(Device *dev, const stl::string &source);
istInterModule PixelShader*    CreatePixelShaderFromString(Device *dev, const stl::string &source);


bool MapAndWrite(DeviceContext *ctx, Buffer *bo, const void *data, size_t data_size);
bool MapAndRead(DeviceContext *ctx, Buffer *bo, void *data, size_t data_size);



istInterModule RenderTarget* CreateRenderTarget(Device *dev, uint32 num_color_buffers, const uvec2 &size,
    I3D_COLOR_FORMAT color_format, uint32 level=0);

istInterModule RenderTarget* CreateRenderTarget(Device *dev, uint32 num_color_buffers, const uvec2 &size,
    I3D_COLOR_FORMAT *color_formats, uint32 level=0);

istInterModule RenderTarget* CreateRenderTarget(Device *dev, uint32 num_color_buffers, const uvec2 &size,
    I3D_COLOR_FORMAT color_format, I3D_COLOR_FORMAT depthstencil_format, uint32 level_color=0, uint32 level_depth=0);

istInterModule RenderTarget* CreateRenderTarget(Device *dev, uint32 num_color_buffers, const uvec2 &size,
    I3D_COLOR_FORMAT *color_formats, I3D_COLOR_FORMAT depthstencil_format, uint32 level_color=0, uint32 level_depth=0);


istInterModule Buffer* CreateVertexBuffer(Device *dev, uint32 size, I3D_USAGE usage, void *data=NULL);
istInterModule Buffer* CreateIndexBuffer(Device *dev, uint32 size, I3D_USAGE usage, void *data=NULL);
istInterModule Buffer* CreateUniformBuffer(Device *dev, uint32 size, I3D_USAGE usage, void *data=NULL);

istInterModule void EnableVSync(bool v);

} // namespace i3d
} // namespace ist
#endif // __ist_i3dgl_Util__
