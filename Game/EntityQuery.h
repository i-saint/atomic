#ifndef atm_Game_EntityQuery_h
#define atm_Game_EntityQuery_h

#include "FunctionID.h"


#define atmECallBlock(...) \
    virtual bool call(FunctionID fid, const void *args, void *ret)\
    {\
        typedef std::remove_reference<decltype(*this)>::type this_t;\
        __VA_ARGS__\
        return false;\
    }

#define atmMethodBlock(...)    \
    switch(fid) {\
    __VA_ARGS__\
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
inline bool atmCallImpl(EntityHandle h, FunctionID fid, const Arg &args)
{
    if(IEntity *e=atmGetEntity(h)) { atmCallImpl(e, fid, args); }
}
template<class Arg, class Ret>
inline bool atmCallImpl(EntityHandle h, FunctionID fid, const Arg &args, Ret &ret)
{
    if(IEntity *e=atmGetEntity(h)) { atmCallImpl(e, fid, args, ret); }
}
template<class C, class Arg>
inline bool atmCallImpl(C *e, FunctionID fid, const Arg &args)
{
    return e->call(fid, &args, NULL);
}
template<class C, class Ret, class Arg>
inline bool atmCallImpl(C *e, FunctionID fid, const Arg &args, Ret &ret)
{
    return e->call(fid, &args, &ret);
}

template<class Ret>
inline bool atmQueryImpl(EntityHandle h, FunctionID fid, Ret &ret)
{
    if(IEntity *e=atmGetEntity(h)) { atmQueryImpl(e, fid, args); }
}
template<class Ret, class Arg>
inline bool atmQueryImpl(EntityHandle h, FunctionID fid, Ret &ret, const Arg &args)
{
    if(IEntity *e=atmGetEntity(h)) { atmQueryImpl(e, fid, ret, args); }
}
template<class C, class Ret>
inline bool atmQueryImpl(C *e, FunctionID fid, Ret &ret)
{
    return e->call(fid, NULL, &ret);
}
template<class C, class Ret, class Arg>
inline bool atmQueryImpl(C *e, FunctionID fid, Ret &ret, const Arg &args)
{
    return e->call(fid, &args, &ret);
}

#define atmArgs(...)                    ist::MakeValueList(__VA_ARGS__)
#define atmCall(entity, funcname, ...)  atmCallImpl(entity, FID_##funcname, __VA_ARGS__)
#define atmQuery(entity, funcname, ...) atmQueryImpl(entity, FID_##funcname, __VA_ARGS__)

} // namespace atm
#endif // atm_Game_EntityQuery_h
