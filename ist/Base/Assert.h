#ifndef __ist_Assert__
#define __ist_Assert__

#ifdef __ist_enable_assert__

#define istPrint(...) ist::DebugPrint(__FILE__, __LINE__, __VA_ARGS__)
#define istAssert(expression, ...)\
    if(!(expression)) {\
        static int s_enable=1;\
        if(s_enable) {\
            s_enable = ist::DebugAssert(__FILE__, __LINE__, __VA_ARGS__);\
        }\
    }

namespace ist
{

    typedef int (*AssertHandler)();
    typedef int (*PanicHandler)();
    istInterModule void SetAssertHandler(AssertHandler handler);
    istInterModule void SetPanicHandler(PanicHandler handler);

    istInterModule void DebugPrint(const char* file, int line, const char* fmt, ...);
    istInterModule void DebugPrintV(const char* file, int line, const char* fmt, va_list vl);
    istInterModule void DebugPrint(const char* file, int line, const wchar_t* fmt, ...);
    istInterModule void DebugPrintV(const char* file, int line, const wchar_t* fmt, va_list vl);

    istInterModule int DebugAssert(const char* file, int line, const char* fmt, ...);
    istInterModule int DebugAssertV(const char* file, int line, const char* fmt, va_list vl);
    istInterModule int DebugAssert(const char* file, int line, const wchar_t* fmt, ...);
    istInterModule int DebugAssertV(const char* file, int line, const wchar_t* fmt, va_list vl);

} // namespace ist

#else // __ist_enable_assert__

#define istPuts(str)
#define istPrint(...)
#define istAssert(...)

#endif // __ist_enable_assert__

#endif // __ist_Assert__
