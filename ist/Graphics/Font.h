#ifndef __ist_Graphics_Font__
#define __ist_Graphics_Font__

#include "GraphicsResource.h"

namespace ist {
namespace graphics {

class SystemFont : GraphicsResource
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

} // namespace graphics
} // namespace ist
#endif __ist_Graphics_Font__
