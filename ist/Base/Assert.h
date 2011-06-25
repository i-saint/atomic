#ifndef __ist_Assert__
#define __ist_Assert__


#ifndef IST_DISABLE_ASSERT

#define IST_PRINT(...) ist::DebugPrint(__FILE__, __LINE__, __VA_ARGS__)
#define IST_ASSERT(...)\
    {\
        static int s_enable=1;\
        if(s_enable) {\
            s_enable = ist::DebugAssert(__FILE__, __LINE__, __VA_ARGS__);\
        }\
    }

namespace ist
{

    typedef int (*AssertHandler)();
    typedef int (*PanicHandler)();
    void SetAssertHandler(AssertHandler handler);
    void SetPanicHandler(PanicHandler handler);

    void DebugPrint(const char* file, int line, const char* fmt, ...);
    void DebugPrintV(const char* file, int line, const char* fmt, va_list vl);

    int DebugAssert(const char* file, int line, const char* fmt, ...);
    int DebugAssertV(const char* file, int line, const char* fmt, va_list vl);

} // namespace ist


#else // IST_DISABLE_ASSERT
#define IST_PRINT(...)
#define IST_ASSERT(...)
#endif // IST_DISABLE_ASSERT

#endif // __ist_Assert__