#ifndef __ist_i3dudx11_Font__
#define __ist_i3dudx11_Font__

#include "i3ddx11DeviceResource.h"

namespace ist {
namespace i3ddx11 {

class SystemFont
{
private:
    HDC m_hdc;
    int m_window_height;
    int m_font_height;

public:
    SystemFont(HDC m_hdc);
    ~SystemFont();

    void draw(int x, int y, const char *text);
    void draw(int x, int y, const wchar_t *text);

    int getFontHeight() const { return m_font_height; }
};

} // namespace i3ddx11
} // namespace ist
#endif __ist_i3dudx11_Font__
