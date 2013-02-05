#include "istPCH.h"
#include "ist/Base.h"
#include <stdarg.h>
#ifdef ist_env_Windows
    #include <tchar.h>
    #include <windows.h>
#endif // ist_env_Windows

#ifdef ist_enable_Assert

namespace ist {


static const int DPRINTF_MES_LENGTH  = 4096;


namespace
{
    AssertHandler s_assert_handler = 0;
    PanicHandler s_panic_handler = 0;
}

#ifdef ist_enable_file_log
namespace
{
    FILE *s_logfile = NULL;
}

void CloseLogFile()
{
    if(s_logfile) {
        fclose(s_logfile);
    }
}

void InitLogFile()
{
    if(!s_logfile) {
        s_logfile = fopen("log.txt", "w");
        if(s_logfile) {
            atexit(CloseLogFile);
        }
    }
}

void WriteLogFile(const char *text)
{
    InitLogFile();
    if(s_logfile) {
        fprintf(s_logfile, text);
        fflush(s_logfile);
    }
}

void WriteLogFile(const wchar_t *text)
{
    InitLogFile();
    if(s_logfile) {
        fwprintf(s_logfile, text);
        fflush(s_logfile);
    }
}
#else // ist_enable_file_log

#define WriteLogFile(...)

#endif // ist_enable_file_log


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
#ifdef ist_env_Windows
    char buf[DPRINTF_MES_LENGTH];
    //istsprintf(buf, "%s:%d - ", file, line);
    //::OutputDebugStringA(buf);
    //WriteLogFile(buf);
    istVSprintf(buf, fmt, vl);
    ::OutputDebugStringA(buf);
    WriteLogFile(buf);
#else // ist_env_Windows
    vprintf(fmt, vl);
    fflush(stdout);
#endif // ist_env_Windows
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
#ifdef ist_env_Windows
    wchar_t wbuf[DPRINTF_MES_LENGTH];
    //char buf[DPRINTF_MES_LENGTH];
    //istsprintf(buf, "%s:%d - ", file, line);
    //::OutputDebugStringA(buf);
    //WriteLogFile(buf);
    istVSprintf(wbuf, fmt, vl);
    ::OutputDebugStringW(wbuf);
    WriteLogFile(wbuf);
#else // ist_env_Windows
    vwprintf(fmt, vl);
    fflush(stdout);
#endif // ist_env_Windows
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
#ifdef ist_env_Windows
    char buf[DPRINTF_MES_LENGTH];
    istSPrintf(buf, "assertion failed %s:%d - ", file, line);
    ::OutputDebugStringA(buf);
    WriteLogFile(buf);
    istVSprintf(buf, fmt, vl);
    ::OutputDebugStringA(buf);
    WriteLogFile(buf);
    DebugBreak();
#else // ist_env_Windows
    printf("assertion failed %s:%d - ", file, line);
    vprintf(fmt, vl);
    fflush(stdout);
    fgets(buf, _countof(buf), stdin);
    return s_assert_handler ? s_assert_handler() : 0;
#endif // ist_env_Windows
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
#ifdef ist_env_Windows
    char buf[DPRINTF_MES_LENGTH];
    wchar_t wbuf[DPRINTF_MES_LENGTH];
    istSPrintf(buf, "assertion failed %s:%d - ", file, line);
    ::OutputDebugStringA(buf);
    WriteLogFile(buf);
    istVSNWPrintf(wbuf, _countof(wbuf), fmt, vl);
    ::OutputDebugStringW(wbuf);
    WriteLogFile(wbuf);
    DebugBreak();
#else // ist_env_Windows
    printf("assertion failed %s:%d - ", file, line);
    vprintf(fmt, vl);
    fflush(stdout);
    fgets(buf, _countof(buf), stdin);
    return s_assert_handler ? s_assert_handler() : 0;
#endif // ist_env_Windows
    return 0;
}


} // namespace ist

#endif // ist_enable_assert
