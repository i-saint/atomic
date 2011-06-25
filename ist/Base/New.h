
template<class T>
inline T* call_destructor(T* p)
{
    if(p) {
        p->~T();
    }
    return p;
}

#define EA_NEW(Type)                new(eastl::get_default_allocator(NULL)->allocate(sizeof(Type)))
#define EA_ALIGNED_NEW(Type, Align) new(eastl::get_default_allocator(NULL)->allocate(sizeof(Type), Align, 0))

#define EA_DELETE(Obj)              {eastl::get_default_allocator(NULL)->deallocate(call_destructor(Obj), sizeof(*Obj)); Obj=NULL;}
