#ifndef ist_Base_ClassInfo_h
#define ist_Base_ClassInfo_h

#include "ist/Config.h"

namespace ist {

    template<class T>
    struct istAPI ClassInfo
    {
        static int getClassID();
        static const char* getClassName();
        static size_t getSize();
        static void* construct();
        static void* destruct(void *p);
        static void* constructPlacement(void *p);
        static void* destructPlacement(void *p);
    };

} // namespace ist

#define istImplementClassInfo(classname)                    \
namespace ist {                                             \
extern int g_classidgen;                                    \
template<> struct ClassInfo<classname>                      \
{                                                           \
    static int s_classid;                                   \
    static int getClassID() { return s_classid; }           \
    static const char* getClassName() { return #classname; }\
    static size_t getSize() { return sizeof(classname); }   \
    static void* construct() { return new classname(); }    \
    static void destruct(void *p) { delete p; }             \
    static void* constructPlacement(void *p) { return new(p) classname(); }                     \
    static void destructPlacement(void *p)  { call_destructor(static_cast<classname*>(p)); }    \
};                                                          \
int ClassInfo<classname>::s_classid=g_classidgen++;         \
}

#endif // ist_Base_ClassInfo_h
