#ifndef ist_Base_New_h
#define ist_Base_New_h

#include "ist/stdex/crtex.h"

// .lib 内に operator delete を定義した場合、CRT の同名シンボルと競合して曖昧なシンボルエラーになってしまう。
// そのため、以下のマクロをアプリケーション側コードのどこかに書いて定義してやる必要がある。
// 引数がやたら多い new[] 2 つは EASTL 用。
#define istImplementOperatorNewDelete()\
    void* operator new(size_t size)                 { return istAlignedMalloc(size, istDefaultAlignment); }\
    void* operator new(size_t size, size_t align)   { return istAlignedMalloc(size, align); }\
    void* operator new[](size_t size)               { return istAlignedMalloc(size, istDefaultAlignment); }\
    void* operator new[](size_t size, size_t align) { return istAlignedMalloc(size, align); }\
    void operator delete(void* p)           { istAlignedFree(p); }\
    void operator delete(void* p, size_t)   { istAlignedFree(p); }\
    void operator delete[](void* p)         { istAlignedFree(p); }\
    void operator delete[](void* p, size_t) { istAlignedFree(p); }\
    void* operator new[](size_t size, const char* pName, int flags, unsigned debugFlags, const char* file, int line)\
    {\
        void* p = istAlignedMalloc(size, istDefaultAlignment);\
        return p;\
    }\
    void* operator new[](size_t size, size_t alignment, size_t alignmentOffset, const char* pName, int flags, unsigned debugFlags, const char* file, int line)\
    {\
        void* p = istAlignedMalloc(size, alignment);\
        return p;\
    }

const size_t istDefaultAlignment = 16;
void* operator new(size_t size);
void* operator new(size_t size, size_t align);
void* operator new[](size_t size);
void* operator new[](size_t size, size_t align);
void operator delete(void *p);
void operator delete(void *p, size_t align);
void operator delete[](void *p);
void operator delete[](void *p, size_t align);


template<class T> inline T& unpointer(T &a) { return a; }
template<class T> inline T& unpointer(T *a) { return *a; }

#define istTypeJoin(...)                    __VA_ARGS__

#define istNew(Type)                        new(istAlignof(Type)) Type
#define istAlignedNew(Type, Align)          new(Align) Type
#define istDelete(Obj)                      delete Obj
#define istSafeDelete(Obj)                  if(Obj){istDelete(Obj); Obj=NULL;}

#define istPlacementNew(Type, Addr)         new (Addr) Type
#define istPlacementDelete(Type, Addr)      ((Type*)(Addr))->~Type();

template<class T> inline T* call_destructor(T* p) { p->~T(); return p; }
#define istMakeDestructable template<class T> friend T* ::call_destructor(T *v)
#define istNewA(Type, Alloc)                new(unpointer(Alloc).allocate(sizeof(Type), istDefaultAlignment))Type
#define istAlignedNewA(Type, Align, Alloc)  new(unpointer(Alloc).allocate(sizeof(Type), Align))Type
#define istDeleteA(Obj, Alloc)              unpointer(Alloc).deallocate(call_destructor(Obj))
#define istSafeDeleteA(Obj, Alloc)          if(Obj){istDeleteA(Obj, Alloc); Obj=NULL;}

#endif // ist_Base_New_h
#include "ist/Base/Allocator.h"
