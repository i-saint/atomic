#include "stdafx.h"
#include "Assert.h"
#include <stdarg.h>
#ifdef _WIN32
    #include <tchar.h>
    #include <windows.h>
#endif // _WIN32

namespace ist {


// Visual Studio のデバッグウィンドウの横幅に対応する文字数(バイト数ではない)
static const int DPRINTF_MES_LENGTH  = 256;


namespace
{
    AssertHandler s_assert_handler = 0;
    PanicHandler s_panic_handler = 0;
}


void SetAssertHandler(AssertHandler handler) { s_assert_handler = handler; }
void SetPanicHandler(PanicHandler handler)   { s_panic_handler = handler; }


void DebugPrint(const char* file, int line, const char* fmt, ...)
{
    va_list vl;
    va_start(vl, fmt);
    DebugPrintV(file, line, fmt, vl);
    va_end(vl);
}

void DebugPrintV(const char* file, int line, const char* fmt, va_list vl)
{
    char buf[DPRINTF_MES_LENGTH];
#ifdef _WIN32
    _snprintf_s(buf, DPRINTF_MES_LENGTH, "%s:%d - ", file, line);
    ::OutputDebugStringA(buf);
    _vsnprintf_s(buf, DPRINTF_MES_LENGTH, fmt, vl);
    ::OutputDebugStringA(buf);
#endif // _WIN32
    vprintf(fmt, vl);
    fflush(stdout);
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
    char buf[DPRINTF_MES_LENGTH];
#ifdef _WIN32
    _snprintf_s(buf, DPRINTF_MES_LENGTH, "assertion failed %s:%d - ", file, line);
    ::OutputDebugStringA(buf);
    _vsnprintf_s(buf, DPRINTF_MES_LENGTH, fmt, vl);
    ::OutputDebugStringA(buf);
    DebugBreak();
#endif // _WIN32
    printf("assert %s : %d\n", file, line);
    vprintf(fmt, vl);

    fflush(stdout);
    fgets(buf, DPRINTF_MES_LENGTH, stdin);
    return s_assert_handler ? s_assert_handler() : 0;
}


} // namespace ist
