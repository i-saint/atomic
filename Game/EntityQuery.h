#ifndef atm_Game_EntityQuery_h
#define atm_Game_EntityQuery_h

#include "FunctionID.h"


#define atmECallBlock(blocks) \
    virtual bool call(FunctionID fid, const void *args, void *ret)\
    {\
        blocks\
        return false;\
    }

#define atmMethodBlock(methods)    \
    switch(fid) {\
    methods\
    }

#define atmECall(funcname)         \
    case FID_##funcname: ist::BinaryCall(&this_t::funcname, *this, ret, args); return true;

#define atmECallSuper(classname)   \
    if(this->classname::call(fid, args, ret)) { return true; }

#define atmECallDelegate(obj)   \
    if(obj && obj->call(fid, args, ret)) {}


namespace atm {



class IEntity;

template<class Arg>
inline bool atmCallImpl(IEntity *e, FunctionID fid, const Arg &args)
{
    return e->call(fid, &args, NULL);
}
template<class Arg, class Ret>
inline bool atmCallImpl(IEntity *e, FunctionID fid, const Arg &args, Ret &ret)
{
    return e->call(fid, &args, &ret);
}

template<class Ret>
inline bool atmQueryImpl(IEntity *e, FunctionID fid, Ret &ret)
{
    return e->call(fid, NULL, &ret);
}
template<class Ret, class Arg>
inline bool atmQueryImpl(IEntity *e, FunctionID fid, Ret &ret, const Arg &args)
{
    return e->call(fid, &args, &ret);
}

#define atmCall(entity, funcname, ...) atmCallImpl(entity, FID_##funcname, __VA_ARGS__)
#define atmQuery(entity, funcname, ...) atmQueryImpl(entity, FID_##funcname, __VA_ARGS__)

} // namespace atm
#endif // atm_Game_EntityQuery_h
