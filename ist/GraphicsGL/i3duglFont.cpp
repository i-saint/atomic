#include "stdafx.h"
#include <wingdi.h>
#include "../Window.h"
#include "i3duglFont.h"


namespace ist {
namespace i3dgl {
#ifdef istWindows


class SystemFont : public IFontRenderer
{
private:
    HDC m_hdc;
    int m_window_height;
    int m_font_height;

public:
    SystemFont(HDC m_hdc);
    ~SystemFont();

    void draw(const vec2 &pos, const char *text);
    void draw(const vec2 &pos, const wchar_t *text);
    void flush();

    float32 getFontHeight() const { return (float32)m_font_height; }
};


static const int g_list_base = 0;

SystemFont::SystemFont(HDC m_hdc)
    : m_hdc(m_hdc)
    , m_window_height(0)
    , m_font_height(0)
{
    SelectObject(m_hdc, GetStockObject(SYSTEM_FONT));
    wglUseFontBitmapsW( m_hdc, 0, 256*32, g_list_base );

    TEXTMETRIC metric;
    GetTextMetrics(m_hdc, &metric);
    m_font_height = metric.tmHeight;
    m_window_height = istGetAplication()->getWindowSize().y;
}

SystemFont::~SystemFont()
{
    m_hdc = NULL;
    m_window_height = 0;
    m_font_height = 0;
}

void SystemFont::draw(const vec2 &pos, const char *text)
{
    int len = strlen(text);
    glWindowPos2i((int32)pos.x, m_window_height-m_font_height-(int32)pos.y);
    glCallLists(len, GL_UNSIGNED_BYTE, text);
}

void SystemFont::draw(const vec2 &pos, const wchar_t *text)
{
    int len = wcslen(text);
    glWindowPos2i((int32)pos.x, m_window_height-m_font_height-(int32)pos.y);
    glCallLists(len, GL_UNSIGNED_SHORT, text);
}

void SystemFont::flush()
{
}

IFontRenderer* CreateSystemFont(void *hdc)
{
    return istNew(SystemFont)((HDC)hdc);
}

#endif // istWindows

} // namespace i3d
} // namespace ist
