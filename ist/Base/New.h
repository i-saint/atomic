#ifndef __ist_Base_New_h__
#define __ist_Base_New_h__

// .lib 内に operator delete を定義した場合、CRT の同名シンボルと競合して曖昧なシンボルエラーになってしまう。
// そのため、以下のマクロをアプリケーション側コードのどこかに書いて定義してやる必要がある。
// (.lib の中だろうとそうでなかろうと競合するのは変わらないはずだが、operator new/delete はリンカからの扱いが特殊なのだと予想される)
// 引数がやたら多い new[] 2 つは EASTL 用。
#define istImplementOperatorNewDelete()\
    void* operator new(size_t size)     { return istRawMalloc(size); }\
    void* operator new[](size_t size)   { return istRawMalloc(size); }\
    void* operator new[](size_t size, const char* pName, int flags, unsigned debugFlags, const char* file, int line)\
    {\
        void* p = istRawMalloc(size, MinimumAlignment);\
        return p;\
    }\
    void* operator new[](size_t size, size_t alignment, size_t alignmentOffset, const char* pName, int flags, unsigned debugFlags, const char* file, int line)\
    {\
        void* p = istRawMalloc(size, alignment);\
        return p;\
    }\
    void operator delete(void* p)   { istRawFree(p); }\
    void operator delete[](void* p) { istRawFree(p); }



template<class T> inline T* call_destructor(T* p) { p->~T(); return p; }
#define istMakeDestructable template<class T> friend T* ::call_destructor(T *v)

const size_t MinimumAlignment = 16;

istInterModule void* istRawMalloc(size_t size, size_t align=MinimumAlignment);
istInterModule void istRawFree(void* p);
istInterModule void* istRawAlloca(size_t size);


template<class T> inline T& unpointer(T &a) { return a; }
template<class T> inline T& unpointer(T *a) { return *a; }

#define istNew(Type)                        new(ist::GetDefaultAllocator()->allocate(sizeof(Type), ist::DefaultAlignment))Type
#define istAlignedNew(Type, Align)          new(ist::GetDefaultAllocator()->allocate(sizeof(Type), Align))Type
#define istDelete(Obj)                      ist::GetDefaultAllocator()->deallocate(call_destructor(Obj))
#define istSafeDelete(Obj)                  if(Obj){istDelete(Obj); Obj=NULL;}
#define istMalloc(Size)                     ist::GetDefaultAllocator()->allocate(Size, ist::DefaultAlignment)
#define istAlignedMalloc(Size, Align)       ist::GetDefaultAllocator()->allocate(Size, Align)
#define istFree(Obj)                        ist::GetDefaultAllocator()->deallocate(Obj)
#define istSafeFree(Obj)                    if(Obj){ist::GetDefaultAllocator()->deallocate(Obj, 0); Obj=NULL;}

#define istNewA(Type, A)                    new(unpointer(A).allocate(sizeof(Type), ist::DefaultAlignment))Type
#define istAlignedNewA(Type, Align, A)      new(unpointer(A).allocate(sizeof(Type), Align))Type
#define istDeleteA(Obj, A)                  unpointer(A).deallocate(call_destructor(Obj))
#define istSafeDeleteA(Obj, A)              if(Obj){istDeleteA(Obj, A); Obj=NULL;}
#define istMallocA(Size, A)                 unpointer(A).allocate(Size, ist::DefaultAlignment)
#define istAlignedMallocA(Size, Align, A)   unpointer(A).allocate(Size, Align)
#define istFreeA(Obj, A)                    unpointer(A).deallocate(Obj)
#define istSafeFreeA(Obj, A)                if(Obj){unpointer(A).deallocate(Obj); Obj=NULL;}


#ifdef __ist_enable_memory_leak_check__
    istInterModule void istMemoryLeakCheckerInitialize();
    istInterModule void istMemoryLeakCheckerFinalize();
    istInterModule void istMemoryLeakCheckerPrint();
    istInterModule void istMemoryLeakCheckerEnable(bool v);
#else // __ist_enable_memory_leak_check__
#   define istMemoryLeakCheckerInitialize()
#   define istMemoryLeakCheckerFinalize()
#   define istMemoryLeakCheckerPrint()
#   define istMemoryLeakCheckerEnable(...)
#endif // __ist_enable_memory_leak_check__

#endif // __ist_Base_New_h__
#include "ist/Base/Allocator.h"
