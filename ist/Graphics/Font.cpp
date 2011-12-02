#include "stdafx.h"
#include <wingdi.h>
#include "../Window/Application.h"
#include "Font.h"


namespace ist {
namespace graphics {
#ifdef WIN32

    static const int g_list_base = 0;

Font::Font()
    : m_hdc(NULL)
    , m_window_height(0)
    , m_font_height(0)
{
}

Font::~Font()
{
}

bool Font::initialize()
{
    m_hdc = istGetAplication()->getHDC();
    SelectObject(m_hdc, GetStockObject(SYSTEM_FONT));
    wglUseFontBitmapsW( m_hdc, 0, 65535, g_list_base );

    TEXTMETRIC metric;
    GetTextMetrics(m_hdc, &metric);
    m_font_height = metric.tmHeight;
    m_window_height = istGetAplication()->getWindowHeight();

    return true;
}

void Font::finalize()
{
    m_hdc = NULL;
    m_window_height = 0;
    m_font_height = 0;
}

void Font::draw(int x, int y, const char *text)
{
    int len = strlen(text);
    glWindowPos2i(x, m_window_height-m_font_height-y);
    glCallLists (len, GL_UNSIGNED_BYTE, text);
}

void Font::draw(int x, int y, const wchar_t *text)
{
    int len = wcslen(text);
    glWindowPos2i(x, m_window_height-m_font_height-y);
    glCallLists (len, GL_UNSIGNED_SHORT, text);
}
#endif // WIN32

} // namespace graphics
} // namespace ist
