#include "stdafx.h"
#include <wingdi.h>
#include "../Window.h"
#include <D3D11.h>
#include <D3DX11.h>
#include "i3dudx11Font.h"


namespace ist {
namespace i3ddx11 {

static const int g_list_base = 0;

SystemFont::SystemFont(HDC m_hdc)
    : m_hdc(m_hdc)
    , m_window_height(0)
    , m_font_height(0)
{
    SelectObject(m_hdc, GetStockObject(SYSTEM_FONT));
}

SystemFont::~SystemFont()
{
    m_hdc = NULL;
    m_window_height = 0;
    m_font_height = 0;
}

void SystemFont::draw(int x, int y, const char *text)
{
    int len = strlen(text);
}

void SystemFont::draw(int x, int y, const wchar_t *text)
{
    int len = wcslen(text);
}

} // namespace i3ddx11
} // namespace ist
