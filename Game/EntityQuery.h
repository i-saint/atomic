#ifndef atomic_Game_EntityQuery_h
#define atomic_Game_EntityQuery_h

#include "FunctionID.h"


#define atomicECall(funcname)         \
    case FID_##funcname: MFCall(*this, &this_t::funcname, v); return true;

#define atomicECallSuper(classname)   \
    if(this->classname::call(call_id, v)) { return true; }

#define atomicECallDelegate(obj)   \
    if(obj && obj->call(call_id, v)) {}

#define atomicEQuery(funcname)    \
    case FID_##funcname: v=funcname(); return true;

#define atomicEQuerySuper(classname)   \
    if(this->classname::query(call_id, v)) { return true; }

#define atomicEQueryDelegate(obj)   \
    if(obj && obj->query(call_id, v)) { return true; }

#define atomicMethodBlock(methods)    \
    switch(call_id) {\
    methods\
    }

#define atomicECallBlock(blocks) \
    virtual bool call(FunctionID call_id, const variant &v)\
    {\
        blocks\
        return false;\
    }

#define atomicEQueryBlock(blocks) \
    virtual bool query(FunctionID call_id, variant &v) const\
    {\
        blocks\
        return false;\
    }


namespace atomic {

template<class T> struct DeRef { typedef T type; };
template<class T> struct DeRef<T&> { typedef T type; };

template<class T, class Res>
inline void MFCall( T &obj, Res (T::*mf)(), const variant &v ) { (obj.*mf)(); }
template<class T, class Res, class Arg1>
inline void MFCall( T &obj, Res (T::*mf)(Arg1), const variant &v ) { (obj.*mf)(v.cast<DeRef<Arg1>::type>()); }


class IEntity;
template<class T>
inline T _atomicQuery(IEntity *e, FunctionID qid)
{
    variant v;
    if(!e->query(qid, v)) {
        istPrint("query failed. entity: 0x%x query: %d\n", e->getHandle(), qid);
    }
    return v.cast<T>();
}

#define atomicCall(entity, funcname, ...) entity->call(FID_##funcname, __VA_ARGS__)
#define atomicQuery(entity, funcname, T) _atomicQuery<T>(entity, FID_##funcname)

} // namespace atomic
#endif // atomic_Game_EntityQuery_h
