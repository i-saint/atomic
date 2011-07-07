
template<class T>
inline T* call_destructor(T* p)
{
    p->~T();
    return p;
}

#define AT_NEW(Type)                    new(stl::get_default_allocator(NULL)->allocate(sizeof(Type)))Type
#define AT_ALIGNED_NEW(Type, Align)     new(stl::get_default_allocator(NULL)->allocate(sizeof(Type), Align, 0))Type
#define AT_DELETE(Obj)                  if(Obj){stl::get_default_allocator(NULL)->deallocate(call_destructor(Obj), sizeof(*Obj)); Obj=NULL;}

#define AT_ALIGNED_MALLOC(Size, Align)  stl::get_default_allocator(NULL)->allocate(Size, Align, 0)
#define AT_FREE(Obj)                    stl::get_default_allocator(NULL)->deallocate(Obj, 0)
