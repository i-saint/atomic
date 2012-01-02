#ifndef __ist_Base_New__
#define __ist_Base_New__

template<class T>
inline T* call_destructor(T* p)
{
    p->~T();
    return p;
}

#define istNew(Type)                    new(stl::get_default_allocator(NULL)->allocate(sizeof(Type)))Type
#define istAlignedNew(Type, Align)      new(stl::get_default_allocator(NULL)->allocate(sizeof(Type), Align, 0))Type
#define istDelete(Obj)                  stl::get_default_allocator(NULL)->deallocate(call_destructor(Obj), sizeof(*Obj))
#define istSafeDelete(Obj)              if(Obj){istDelete(Obj); Obj=NULL;}

#define istMalloc(Size)                 stl::get_default_allocator(NULL)->allocate(Size)
#define istAlignedMalloc(Size, Align)   stl::get_default_allocator(NULL)->allocate(Size, Align, 0)
#define istFree(Obj)                    stl::get_default_allocator(NULL)->deallocate(Obj, 0)
#define istSafeFree(Obj)                if(Obj){stl::get_default_allocator(NULL)->deallocate(Obj, 0); Obj=NULL;}

#define istSafeRelease(Obj)             if(Obj){Obj->release();Obj=NULL;}
#define istSafeAddRef(Obj)              if(Obj){Obj->addRef();}

#endif // __ist_Base_New__
