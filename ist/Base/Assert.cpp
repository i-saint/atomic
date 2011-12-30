#include "stdafx.h"
#include "Assert.h"
#include <stdarg.h>
#ifdef _WIN32
    #include <tchar.h>
    #include <windows.h>
#endif // _WIN32

#ifdef __ist_enable_assert__

namespace ist {


static const int DPRINTF_MES_LENGTH  = 1024;


namespace
{
    AssertHandler s_assert_handler = 0;
    PanicHandler s_panic_handler = 0;
}


void SetAssertHandler(AssertHandler handler) { s_assert_handler = handler; }
void SetPanicHandler(PanicHandler handler)   { s_panic_handler = handler; }


void DebugPuts(const char* str)
{
#ifdef _WIN32
    ::OutputDebugStringA(str);
#else
    puts(str);
    fflush(stdout);
#endif // _WIN32
}

void DebugPuts(const wchar_t* str)
{
#ifdef _WIN32
    ::OutputDebugStringW(str);
#else
    puts(str);
    fflush(stdout);
#endif // _WIN32
}

void DebugPrint(const char* file, int line, const char* fmt, ...)
{
    va_list vl;
    va_start(vl, fmt);
    DebugPrintV(file, line, fmt, vl);
    va_end(vl);
}

void DebugPrintV(const char* file, int line, const char* fmt, va_list vl)
{
#ifdef _WIN32
    char buf[DPRINTF_MES_LENGTH];
    _snprintf_s(buf, DPRINTF_MES_LENGTH, "%s:%d - ", file, line);
    ::OutputDebugStringA(buf);
    _vsnprintf_s(buf, DPRINTF_MES_LENGTH, fmt, vl);
    ::OutputDebugStringA(buf);
#else
    vprintf(fmt, vl);
    fflush(stdout);
#endif // _WIN32
}

void DebugPrint(const char* file, int line, const wchar_t* fmt, ...)
{
    va_list vl;
    va_start(vl, fmt);
    DebugPrintV(file, line, fmt, vl);
    va_end(vl);

}
void DebugPrintV(const char* file, int line, const wchar_t* fmt, va_list vl)
{
#ifdef _WIN32
    char buf[DPRINTF_MES_LENGTH];
    wchar_t wbuf[DPRINTF_MES_LENGTH];
    _snprintf_s(buf, DPRINTF_MES_LENGTH, "%s:%d - ", file, line);
    ::OutputDebugStringA(buf);
    _vsnwprintf_s(wbuf, DPRINTF_MES_LENGTH, _TRUNCATE, fmt, vl);
    ::OutputDebugStringW(wbuf);
#else
    vwprintf(fmt, vl);
    fflush(stdout);
#endif // _WIN32
}


int DebugAssert(const char* file, int line, const char* fmt, ...)
{
    va_list vl;
    va_start(vl, fmt);
    int result = DebugAssertV(file, line, fmt, vl);
    va_end(vl);
    return result;
}

int DebugAssertV(const char* file, int line, const char* fmt, va_list vl)
{
#ifdef _WIN32
    char buf[DPRINTF_MES_LENGTH];
    _snprintf_s(buf, DPRINTF_MES_LENGTH, "assertion failed %s:%d - ", file, line);
    ::OutputDebugStringA(buf);
    _vsnprintf_s(buf, DPRINTF_MES_LENGTH, fmt, vl);
    ::OutputDebugStringA(buf);
    DebugBreak();
#else
    printf("assertion failed %s:%d - ", file, line);
    vprintf(fmt, vl);
    fflush(stdout);
    fgets(buf, DPRINTF_MES_LENGTH, stdin);
    return s_assert_handler ? s_assert_handler() : 0;
#endif // _WIN32
    return 0;
}

int DebugAssert(const char* file, int line, const wchar_t* fmt, ...)
{
    va_list vl;
    va_start(vl, fmt);
    int result = DebugAssertV(file, line, fmt, vl);
    va_end(vl);
    return result;
}

int DebugAssertV(const char* file, int line, const wchar_t* fmt, va_list vl)
{
#ifdef _WIN32
    char buf[DPRINTF_MES_LENGTH];
    wchar_t wbuf[DPRINTF_MES_LENGTH];
    _snprintf_s(buf, DPRINTF_MES_LENGTH, "assertion failed %s:%d - ", file, line);
    ::OutputDebugStringA(buf);
    _vsnwprintf_s(wbuf, DPRINTF_MES_LENGTH, _TRUNCATE, fmt, vl);
    ::OutputDebugStringW(wbuf);
    DebugBreak();
#else
    printf("assertion failed %s:%d - ", file, line);
    vprintf(fmt, vl);
    fflush(stdout);
    fgets(buf, DPRINTF_MES_LENGTH, stdin);
    return s_assert_handler ? s_assert_handler() : 0;
#endif // _WIN32
    return 0;
}


} // namespace ist

#endif // __ist_enable_assert__
