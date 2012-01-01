#ifndef __ist_i3du_Font__
#define __ist_i3du_Font__

#include "i3dDeviceResource.h"

namespace ist {
namespace i3d {

class SystemFont : DeviceResource
{
private:
#ifdef _WIN32
    HDC m_hdc;
#endif // _WIN32
    int m_window_height;
    int m_font_height;

public:
    SystemFont();
    ~SystemFont();

    bool initialize();
    void finalize();

    void draw(int x, int y, const char *text);
    void draw(int x, int y, const wchar_t *text);

    int getFontHeight() const { return m_font_height; }
};

} // namespace i3d
} // namespace ist
#endif __ist_i3du_Font__
