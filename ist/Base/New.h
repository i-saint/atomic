
template<class T>
inline T* call_destructor(T* p)
{
    p->~T();
    return p;
}

#define IST_NEW(Type)                   new(stl::get_default_allocator(NULL)->allocate(sizeof(Type)))Type
#define IST_ALIGNED_NEW(Type, Align)    new(stl::get_default_allocator(NULL)->allocate(sizeof(Type), Align, 0))Type
#define IST_NEW16(Type)                 IST_ALIGNED_NEW(Type,16)
#define IST_DELETE(Obj)                 stl::get_default_allocator(NULL)->deallocate(call_destructor(Obj), sizeof(*Obj))
#define IST_SAFE_DELETE(Obj)            if(Obj){IST_DELETE(Obj); Obj=NULL;}

#define IST_MALLOC(Size)                stl::get_default_allocator(NULL)->allocate(Size)
#define IST_ALIGNED_MALLOC(Size, Align) stl::get_default_allocator(NULL)->allocate(Size, Align, 0)
#define IST_FREE(Obj)                   stl::get_default_allocator(NULL)->deallocate(Obj, 0)

#define IST_MUST_BE_DELETED(Obj)        if(Obj) { IST_DELETE(Obj);  IST_PRINT("deleted: %p\n", Obj); Obj=NULL; }
