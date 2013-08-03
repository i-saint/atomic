#ifndef ist_i3dgl_Util_h
#define ist_i3dgl_Util_h

#include "i3dglShader.h"
#include "i3dglRenderTarget.h"

class SFMT;

namespace ist {
namespace i3dgl {

// 画像ファイル/ストリームからテクスチャ生成
istAPI Texture2D* CreateTexture2DFromFile(Device *dev, const char *filename, I3D_COLOR_FORMAT format=I3D_COLOR_UNKNOWN);
istAPI Texture2D* CreateTexture2DFromStream(Device *dev, IBinaryStream &st, I3D_COLOR_FORMAT format=I3D_COLOR_UNKNOWN);
istAPI Texture2D* CreateTexture2DFromImage(Device *dev, Image &img, I3D_COLOR_FORMAT format=I3D_COLOR_UNKNOWN);

// 乱数テクスチャ生成
istAPI Texture2D* GenerateRandomTexture(Device *dev, const uvec2 &size, I3D_COLOR_FORMAT format);
istAPI Texture2D* GenerateRandomTexture(Device *dev, const uvec2 &size, I3D_COLOR_FORMAT format, SFMT& random);

// ファイル/ストリームから各種シェーダ生成
istAPI VertexShader*   CreateVertexShaderFromFile(Device *dev, const char *filename);
istAPI GeometryShader* CreateGeometryShaderFromFile(Device *dev, const char *filename);
istAPI PixelShader*    CreatePixelShaderFromFile(Device *dev, const char *filename);

istAPI VertexShader*   CreateVertexShaderFromStream(Device *dev, IBinaryStream &st);
istAPI GeometryShader* CreateGeometryShaderFromStream(Device *dev, IBinaryStream &st);
istAPI PixelShader*    CreatePixelShaderFromStream(Device *dev, IBinaryStream &st);

istAPI VertexShader*   CreateVertexShaderFromString(Device *dev, const stl::string &source);
istAPI GeometryShader* CreateGeometryShaderFromString(Device *dev, const stl::string &source);
istAPI PixelShader*    CreatePixelShaderFromString(Device *dev, const stl::string &source);


bool MapAndWrite(DeviceContext *ctx, Buffer *bo, const void *data, size_t data_size);
bool MapAndRead(DeviceContext *ctx, Buffer *bo, void *data, size_t data_size);



istAPI RenderTarget* CreateRenderTarget(Device *dev, uint32 num_color_buffers, const uvec2 &size,
    I3D_COLOR_FORMAT color_format, uint32 level=0);

istAPI RenderTarget* CreateRenderTarget(Device *dev, uint32 num_color_buffers, const uvec2 &size,
    I3D_COLOR_FORMAT *color_formats, uint32 level=0);

istAPI RenderTarget* CreateRenderTarget(Device *dev, uint32 num_color_buffers, const uvec2 &size,
    I3D_COLOR_FORMAT color_format, I3D_COLOR_FORMAT depthstencil_format, uint32 level_color=0, uint32 level_depth=0);

istAPI RenderTarget* CreateRenderTarget(Device *dev, uint32 num_color_buffers, const uvec2 &size,
    I3D_COLOR_FORMAT *color_formats, I3D_COLOR_FORMAT depthstencil_format, uint32 level_color=0, uint32 level_depth=0);


istAPI Buffer* CreateVertexBuffer(Device *dev, uint32 size, I3D_USAGE usage, void *data=NULL);
istAPI Buffer* CreateIndexBuffer(Device *dev, uint32 size, I3D_USAGE usage, void *data=NULL);
istAPI Buffer* CreateUniformBuffer(Device *dev, uint32 size, I3D_USAGE usage, void *data=NULL);

istAPI void EnableVSync(bool v);

} // namespace i3d
} // namespace ist
#endif // ist_i3dgl_Util_h
