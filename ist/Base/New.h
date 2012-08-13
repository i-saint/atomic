#ifndef __ist_Base_New__
#define __ist_Base_New__

template<class T> inline T* call_destructor(T* p) { p->~T(); return p; }
istInterModule void* istnew(size_t size);
istInterModule void istdelete(void* p);

void* operator new[](size_t size);
void operator delete[](void* p);


#define istImplementNew()\
    void* operator new[](size_t size) { return istnew(size); }

#define istImplementDelete()\
    void operator delete[](void* p) { istdelete(p); }

#define istNew(Type)                    new(stl::get_default_allocator(NULL)->allocate(sizeof(Type)))Type
#define istAlignedNew(Type, Align)      new(stl::get_default_allocator(NULL)->allocate(sizeof(Type), Align, 0))Type
#define istDelete(Obj)                  stl::get_default_allocator(NULL)->deallocate(call_destructor(Obj), sizeof(*Obj))
#define istSafeDelete(Obj)              if(Obj){istDelete(Obj); Obj=NULL;}
#define istMalloc(Size)                 stl::get_default_allocator(NULL)->allocate(Size)
#define istAlignedMalloc(Size, Align)   stl::get_default_allocator(NULL)->allocate(Size, Align, 0)
#define istFree(Obj)                    stl::get_default_allocator(NULL)->deallocate(Obj, 0)
#define istSafeFree(Obj)                if(Obj){stl::get_default_allocator(NULL)->deallocate(Obj, 0); Obj=NULL;}

#define istNewA(Type, A)                    new(A.allocate(sizeof(Type)))Type
#define istAlignedNewA(Type, Align, A)      new(A.allocate(sizeof(Type), Align, 0))Type
#define istDeleteA(Obj, A)                  A.deallocate(call_destructor(Obj), sizeof(*Obj))
#define istSafeDeleteA(Obj, A)              if(Obj){istDeleteA(Obj, A); Obj=NULL;}
#define istMallocA(Size, A)                 A.allocate(Size)
#define istAlignedMallocA(Size, Align, A)   A.allocate(Size, Align, 0)
#define istFreeA(Obj, A)                    A.deallocate(Obj, 0)
#define istSafeFreeA(Obj, A)                if(Obj){A.deallocate(Obj, 0); Obj=NULL;}


#define istSafeRelease(Obj)             if(Obj){Obj->release();Obj=NULL;}
#define istSafeAddRef(Obj)              if(Obj){Obj->addRef();}


#endif // __ist_Base_New__
