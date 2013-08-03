#ifndef ist_Base_Assert_h
#define ist_Base_Assert_h

#ifdef ist_enable_Assert

#define istPrint(...) ist::DebugPrint(__FILE__, __LINE__, __VA_ARGS__)
#define istAssert(expression)\
    if(!(expression)) {\
        static int s_enable=1;\
        if(s_enable) {\
            s_enable = ist::DebugAssert(__FILE__, __LINE__, #expression);\
        }\
    }

namespace ist {

    typedef int (*AssertHandler)();
    typedef int (*PanicHandler)();
    istAPI void SetAssertHandler(AssertHandler handler);
    istAPI void SetPanicHandler(PanicHandler handler);

    istAPI void DebugPrint(const char* file, int line, const char* fmt, ...);
    istAPI void DebugPrintV(const char* file, int line, const char* fmt, va_list vl);
    istAPI void DebugPrint(const char* file, int line, const wchar_t* fmt, ...);
    istAPI void DebugPrintV(const char* file, int line, const wchar_t* fmt, va_list vl);

    istAPI int DebugAssert(const char* file, int line, const char* fmt, ...);
    istAPI int DebugAssertV(const char* file, int line, const char* fmt, va_list vl);

} // namespace ist

#else // ist_enable_assert

#define istPuts(str)
#define istPrint(...)
#define istAssert(...)

#endif // ist_enable_assert

#endif // ist_Base_Assert_h
