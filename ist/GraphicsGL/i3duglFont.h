#ifndef ist_i3dugl_Font_h
#define ist_i3dugl_Font_h

#include "i3dglDeviceResource.h"

namespace ist {
namespace i3dgl {

class IFontRenderer : public SharedObject
{
public:
    IFontRenderer();
    virtual void setScreen(float32 left, float32 right, float32 bottom, float32 top)=0;
    virtual void setColor(const vec4 &v)=0;
    virtual void setSize(float32 v)=0;
    virtual void setSpacing(float32 v)=0; // 文字幅の倍率
    virtual void setMonospace(bool v)=0; // 等幅にするか

    virtual void addText(const vec2 &pos, const char *text, size_t len=0)=0;      // len==0 だと strlen で自動的に計算します
    virtual void addText(const vec2 &pos, const wchar_t *text, size_t len=0)=0;   // wchar_t 版。こっちの方が速いのでできるだけこっち使いましょう
    virtual vec2 computeTextSize(const char *text, size_t len=0)=0;
    virtual vec2 computeTextSize(const wchar_t *text, size_t len=0)=0;
    virtual void draw()=0;
};

class EasyDrawer;
IFontRenderer* CreateSpriteFont(const char *path_to_sff, const char *path_to_img, EasyDrawer *drawer=NULL);
IFontRenderer* CreateSpriteFont(IBinaryStream &sff, IBinaryStream &img, EasyDrawer *drawer=NULL);

} // namespace i3d
} // namespace ist
#endif ist_i3dugl_Font_h
