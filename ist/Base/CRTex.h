#ifndef __ist_Base_CRTex_h__
#define __ist_Base_CRTex_h__

#ifdef istWindows

#define istsnprintf(buf, count, format, ...)    _snprintf(buf, count, format, __VA_ARGS__)
#define istsnwprintf(buf, count, format, ...)   _snwprintf(buf, count, format, __VA_ARGS__)
#define istvsnprintf(buf, count, format, va)    _vsnprintf(buf, count, format, va)
#define istvsnwprintf(buf, count, format, va)   _vsnwprintf(buf, count, format, va)

#else // istWindows

#define istsnprintf(buf, count, format, ...)    snprintf(buf, count, format, __VA_ARGS__)
#define istsnwprintf(buf, count, format, ...)   snwprintf(buf, count, format, __VA_ARGS__)
#define istvsnprintf(buf, count, format, va)    vsnprintf(buf, count, format, va)
#define istvsnwprintf(buf, count, format, va)   vsnwprintf(buf, count, format, va)

#endif // istWindows

template<size_t N>
inline int istsprintf(char (&buf)[N], const char *format, ...)
{
    va_list vl;
    va_start(vl, format);
    int r = istvsnprintf(buf, N, format, vl);
    va_end(vl);
    return r;
}

template<size_t N>
inline int istsprintf(wchar_t (&buf)[N], const wchar_t *format, ...)
{
    va_list vl;
    va_start(vl, format);
    int r = istvsnwprintf(buf, N, format, vl);
    va_end(vl);
    return r;
}

template<size_t N>
inline int istvsprintf(char (&buf)[N], const char *format, va_list vl)
{
    return istvsnprintf(buf, N, format, vl);
}

template<size_t N>
inline int istvsprintf(wchar_t (&buf)[N], const wchar_t *format, va_list vl)
{
    return istvsnwprintf(buf, N, format, vl);
}

#endif // __ist_Base_CRTex_h__
