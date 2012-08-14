#ifndef __ist_i3dugl_Font__
#define __ist_i3dugl_Font__

#include "i3dglDeviceResource.h"

namespace ist {
namespace i3dgl {


class IFontRenderer : public SharedObject
{
public:
    virtual void addText(const vec2 &pos, const char *text, size_t len, float32 size)=0;
    virtual void addText(const vec2 &pos, const wchar_t *text, size_t len, float32 size)=0;
    virtual void flush()=0;
};

IFontRenderer* CreateSystemFont(Device *device, void *hdc);
IFontRenderer* CreateSpriteFont(Device *device, const char *path_to_sff, const char *path_to_png);

} // namespace i3d
} // namespace ist
#endif __ist_i3dugl_Font__
