#include "istPCH.h"
#include "StringSymbol.h"

namespace ist {

stl::wstring L(const stl::string &mbs)
{
    if(mbs.empty()) { return stl::wstring(); }
    size_t wlen = mbstowcs(nullptr, mbs.c_str(), 0);
    if(wlen==size_t(-1)) { return stl::wstring(); }

    stl::wstring wtext;
    wtext.resize(wlen);
    mbstowcs(&wtext[0], mbs.c_str(), wlen);
    return wtext;
}

stl::wstring L(const char *mbs, size_t len=0)
{
    if(len==0) { len=strlen(mbs); }
    return L(stl::string(mbs, len));
}

stl::string S(const std::wstring &wcs)
{
    if(wcs.empty()) { return stl::string(); }
    size_t len = wcstombs(nullptr, wcs.c_str(), 0);
    if(len==size_t(-1)) { return stl::string(); }

    stl::string text;
    text.resize(len);
    wcstombs(&text[0], wcs.c_str(), len);
    return text;
}

stl::string S(const wchar_t *wcs, size_t len=0)
{
    if(len==0) { len=wcslen(wcs); }
    return S(std::wstring(wcs, len));
}



StringSymbolPool::StringSymbolPool() : m_idgen(0) {}
StringSymbolPool::id_t StringSymbolPool::genID(const stl::string &str)
{
    StringTable::iterator i = m_table.find(str);
    if(i==m_table.end()) {
        id_t id = ++m_idgen;
        m_table[str] = id;
        return id;
    }
    return i->second;
}

} // namespace ist

