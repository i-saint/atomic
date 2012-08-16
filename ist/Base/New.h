#ifndef __ist_Base_New__
#define __ist_Base_New__

template<class T> inline T* call_destructor(T* p) { p->~T(); return p; }
#define istMakeDestructable template<class T> friend T* ::call_destructor(T *v)

const size_t MinimumAlignment = 16;

istInterModule void* istRawMalloc(size_t size, size_t align=MinimumAlignment);
istInterModule void istRawFree(void* p);
istInterModule void* istRawAlloca(size_t size);

void* operator new[](size_t size);
void* operator new[](size_t size, const char* pName, int flags, unsigned debugFlags, const char* file, int line);
void* operator new[](size_t size, size_t alignment, size_t alignmentOffset, const char* pName, int flags, unsigned debugFlags, const char* file, int line);
void operator delete[](void* p);

#define istImplementNew()\
    void* operator new[](size_t size)\
    {\
        return istRawMalloc(size);\
    }\
    void* operator new[](size_t size, const char* pName, int flags, unsigned debugFlags, const char* file, int line)\
    {\
        void* p = istRawMalloc(size, MinimumAlignment);\
        return p;\
    }\
    void* operator new[](size_t size, size_t alignment, size_t alignmentOffset, const char* pName, int flags, unsigned debugFlags, const char* file, int line)\
    {\
        void* p = istRawMalloc(size, alignment);\
        return p;\
    }

#define istImplementDelete()\
    void operator delete[](void* p) { istRawFree(p); }



#define istNew(Type)                    new(ist::GetDefaultAllocator()->allocate(sizeof(Type), ist::DefaultAlignment))Type
#define istAlignedNew(Type, Align)      new(ist::GetDefaultAllocator()->allocate(sizeof(Type), Align))Type
#define istDelete(Obj)                  ist::GetDefaultAllocator()->deallocate(call_destructor(Obj))
#define istSafeDelete(Obj)              if(Obj){istDelete(Obj); Obj=NULL;}
#define istMalloc(Size)                 ist::GetDefaultAllocator()->allocate(Size, ist::DefaultAlignment)
#define istAlignedMalloc(Size, Align)   ist::GetDefaultAllocator()->allocate(Size, Align)
#define istFree(Obj)                    ist::GetDefaultAllocator()->deallocate(Obj)
#define istSafeFree(Obj)                if(Obj){ist::GetDefaultAllocator()->deallocate(Obj, 0); Obj=NULL;}

#define istNewA(Type, A)                    new(A.allocate(sizeof(Type), ist::DefaultAlignment))Type
#define istAlignedNewA(Type, Align, A)      new(A.allocate(sizeof(Type), Align))Type
#define istDeleteA(Obj, A)                  A.deallocate(call_destructor(Obj))
#define istSafeDeleteA(Obj, A)              if(Obj){istDeleteA(Obj, A); Obj=NULL;}
#define istMallocA(Size, A)                 A.allocate(Size, ist::DefaultAlignment)
#define istAlignedMallocA(Size, Align, A)   A.allocate(Size, Align)
#define istFreeA(Obj, A)                    A.deallocate(Obj)
#define istSafeFreeA(Obj, A)                if(Obj){A.deallocate(Obj); Obj=NULL;}


#endif // __ist_Base_New__
#include "ist/Base/Allocator.h"
