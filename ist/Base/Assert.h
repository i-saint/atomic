#ifndef ist_Base_Assert_h
#define ist_Base_Assert_h

#ifdef ist_enable_assert

#define istPrint(...) ist::DebugPrint(__FILE__, __LINE__, __VA_ARGS__)
#define istAssert(expression)\
    if(!(expression)) {\
        static int s_enable=1;\
        if(s_enable) {\
            s_enable = ist::DebugAssert(__FILE__, __LINE__, #expression);\
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

#else // ist_enable_assert

#define istPuts(str)
#define istPrint(...)
#define istAssert(...)

#endif // ist_enable_assert

#endif // ist_Base_Assert_h
