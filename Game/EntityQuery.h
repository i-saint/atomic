#ifndef atomic_Game_EntityQuery_h
#define atomic_Game_EntityQuery_h

#include "FunctionID.h"


#define atomicECallBlock(blocks) \
    virtual bool call(FunctionID fid, const void *args, void *ret)\
    {\
        blocks\
        return false;\
    }

#define atomicMethodBlock(methods)    \
    switch(fid) {\
    methods\
    }

#define atomicECall(funcname)         \
    case FID_##funcname: ist::BinaryCall(&this_t::funcname, *this, ret, args); return true;

#define atomicECallSuper(classname)   \
    if(this->classname::call(fid, args, ret)) { return true; }

#define atomicECallDelegate(obj)   \
    if(obj && obj->call(fid, args, ret)) {}


namespace atomic {



class IEntity;

template<class Arg>
inline bool atomicCallImpl(IEntity *e, FunctionID fid, const Arg &args)
{
    return e->call(fid, &args, NULL);
}
template<class Arg, class Ret>
inline bool atomicCallImpl(IEntity *e, FunctionID fid, const Arg &args, Ret &ret)
{
    return e->call(fid, &args, &ret);
}

template<class Ret>
inline bool atomicQueryImpl(IEntity *e, FunctionID fid, Ret &ret)
{
    return e->call(fid, NULL, &ret);
}
template<class Ret, class Arg>
inline bool atomicQueryImpl(IEntity *e, FunctionID fid, Ret &ret, const Arg &args)
{
    return e->call(fid, &args, &ret);
}

#define atomicCall(entity, funcname, ...) atomicCallImpl(entity, FID_##funcname, __VA_ARGS__)
#define atomicQuery(entity, funcname, ...) atomicQueryImpl(entity, FID_##funcname, __VA_ARGS__)

} // namespace atomic
#endif // atomic_Game_EntityQuery_h
