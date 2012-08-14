#ifndef __ist_i3dugl_Font__
#define __ist_i3dugl_Font__

#include "i3dglDeviceResource.h"

namespace ist {
namespace i3dgl {


class IFontRenderer : public SharedObject
{
public:
    virtual void draw(const vec2 &pos, const char *text)=0;
    virtual void draw(const vec2 &pos, const wchar_t *text)=0;
    virtual void flush()=0;
    virtual float32 getFontHeight() const=0;
};

IFontRenderer* CreateSystemFont(void *hdc);
IFontRenderer* CreateTTFFont(const char *path_to_ttf, float32 font_size);

} // namespace i3d
} // namespace ist
#endif __ist_i3dugl_Font__
