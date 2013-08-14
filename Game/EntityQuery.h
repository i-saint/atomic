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

template<class C>
inline bool atmCallImpl(C *e, FunctionID fid)
{
    return e->call(fid, nullptr, nullptr);
}
template<class C, class Arg>
inline bool atmCallImpl(C *e, FunctionID fid, const Arg &args)
{
    return e->call(fid, &args, nullptr);
}
template<class C, class Ret, class Arg>
inline bool atmCallImpl(C *e, FunctionID fid, const Arg &args, Ret &ret)
{
    return e->call(fid, &args, &ret);
}

inline bool atmCallImpl(EntityHandle h, FunctionID fid)
{
    if(IEntity *e=atmGetEntity(h)) { return atmCallImpl(e, fid); }
    return false;
}
template<class Arg>
inline bool atmCallImpl(EntityHandle h, FunctionID fid, const Arg &args)
{
    if(IEntity *e=atmGetEntity(h)) { return atmCallImpl(e, fid, args); }
    return false;
}
template<class Arg, class Ret>
inline bool atmCallImpl(EntityHandle h, FunctionID fid, const Arg &args, Ret &ret)
{
    if(IEntity *e=atmGetEntity(h)) { return atmCallImpl(e, fid, args, ret); }
    return false;
}

template<class Ret>
inline bool atmQueryImpl(EntityHandle h, FunctionID fid, Ret &ret)
{
    if(IEntity *e=atmGetEntity(h)) { return atmQueryImpl(e, fid, ret); }
    return false;
}
template<class Ret, class Arg>
inline bool atmQueryImpl(EntityHandle h, FunctionID fid, Ret &ret, const Arg &args)
{
    if(IEntity *e=atmGetEntity(h)) {return atmQueryImpl(e, fid, ret, args); }
    return false;
}
template<class C, class Ret>
inline bool atmQueryImpl(C *e, FunctionID fid, Ret &ret)
{
    return e->call(fid, nullptr, &ret);
}
template<class C, class Ret, class Arg>
inline bool atmQueryImpl(C *e, FunctionID fid, Ret &ret, const Arg &args)
{
    return e->call(fid, &args, &ret);
}

template<class T>
inline T atmGetProperyImpl(EntityHandle h, FunctionID fid)
{
    T ret = T();
    if(IEntity *e=atmGetEntity(h)) { e->call(fid, nullptr, &ret); }
    return ret;
}
template<class T>
inline T atmGetProperyImpl(IEntity *e, FunctionID fid)
{
    T ret = T();
    if(e) { e->call(fid, nullptr, &ret); }
    return ret;
}


template<class T>
class EntityProperty
{
public:
    EntityProperty(EntityHandle h, FunctionID getter, FunctionID setter=FID_unknown)
        : m_handle(h), m_getter(getter), m_setter(setter)
    {}

    EntityProperty(IEntity *e, FunctionID getter, FunctionID setter=FID_unknown)
        : m_handle(e ? e->getHandle() : 0), m_getter(getter), m_setter(setter)
    {}

    bool get(T &v)
    {
        if(IEntity *e=atmGetEntity(m_handle)) {
            if(e->call(m_getter, nullptr, &v)) {
                return true;
            }
        }
        else {
            m_handle = 0;
        }
        return false;
    }

    bool set(const T &v)
    {
        if(IEntity *e=atmGetEntity(m_handle)) {
            if(e->call(m_setter, &v, nullptr)) {
                return true;
            }
        }
        else {
            m_handle = 0;
        }
        return false;
    }

    operator T()
    {
        T tmp = T();
        get(&tmp);
        return tmp;
    }

    void operator=(const T &v)
    {
        set(v);
    }

private:
    EntityHandle m_handle;
    FunctionID m_getter;
    FunctionID m_setter;
};

#define atmArgs(...)                    ist::MakeValueList(__VA_ARGS__)
#define atmCall(entity, funcname, ...)  atmCallImpl(entity, FID_##funcname, __VA_ARGS__)
#define atmCall1(entity, funcname)  atmCallImpl(entity, FID_##funcname)
#define atmQuery(entity, funcname, ...) atmQueryImpl(entity, FID_##funcname, __VA_ARGS__)
#define atmGetProperty(type, entity, funcname, ...) atmGetProperyImpl<type>(entity, FID_##funcname)

} // namespace atm
#endif // atm_Game_EntityQuery_h
