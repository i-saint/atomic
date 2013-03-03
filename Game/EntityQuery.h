#ifndef atomic_Game_EntityQuery_h
#define atomic_Game_EntityQuery_h

#include "FunctionID.h"


#define atomicECallBlock(blocks) \
    virtual bool call(FunctionID fid, const variant *args, variant *ret)\
    {\
        blocks\
        return false;\
    }

#define atomicMethodBlock(methods)    \
    switch(fid) {\
    methods\
    }

#define atomicECall(funcname)         \
    case FID_##funcname: ist::VariantCall(&this_t::funcname, *this, ret, args); return true;

#define atomicECallSuper(classname)   \
    if(this->classname::call(fid, args, ret)) { return true; }

#define atomicECallDelegate(obj)   \
    if(obj && obj->call(fid, args, ret)) {}


namespace atomic {


template<class T, class Res>
inline void MFCall( T &obj, Res (T::*mf)(), const variant &v ) { (obj.*mf)(); }
template<class T, class Res, class Arg1>
inline void MFCall( T &obj, Res (T::*mf)(Arg1), const variant &v ) { (obj.*mf)(v.cast<std::remove_reference<Arg1>::type>()); }


class IEntity;
template<class T>
inline T _atomicQuery(IEntity *e, FunctionID fid, const variant *args=NULL)
{
    variant v;
    if(!e->call(fid, args, &v)) {
        //istPrint("query failed. entity: 0x%x query: %d\n", e->getHandle(), qid);
    }
    return v.cast<T>();
}

#define atomicCall(entity, funcname, ...) entity->call(FID_##funcname, __VA_ARGS__)
#define atomicQuery(entity, funcname, T, ...) _atomicQuery<T>(entity, FID_##funcname, __VA_ARGS__)

} // namespace atomic
#endif // atomic_Game_EntityQuery_h
