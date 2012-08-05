#ifndef __atomic_Game_EntityQuery__
#define __atomic_Game_EntityQuery__



#define atomicECall(funcname)         \
    case ECALL_##funcname: MFCall(*this, &this_t::funcname, v); return true;

#define atomicECallSuper(classname)   \
    if(this->classname::call(call_id, v)) { return true; }

#define atomicECallDelegate(obj)   \
    if(obj && obj->call(call_id, v)) {}

#define atomicEQuery(funcname)    \
    case EQUERY_##funcname: v=funcname(); return true;

#define atomicEQuerySuper(classname)   \
    if(this->classname::query(call_id, v)) { return true; }

#define atomicEQueryDelegate(obj)   \
    if(obj && obj->query(call_id, v)) { return true; }

#define atomicMethodBlock(methods)    \
    switch(call_id) {\
    methods\
    }

#define atomicECallBlock(blocks) \
    virtual bool call(uint32 call_id, const variant &v)\
    {\
        blocks\
        return false;\
    }

#define atomicEQueryBlock(blocks) \
    virtual bool query(uint32 call_id, variant &v) const\
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


enum ENTITY_CALL
{
    ECALL_kill,
    ECALL_destroy,
    ECALL_setRefCount,
    ECALL_addRefCount,
    ECALL_release,
    ECALL_eventCollide,
    ECALL_eventFluid,
    ECALL_eventDamage,
    ECALL_eventDestroy,
    ECALL_eventKill,
    ECALL_damage,
    ECALL_setDiffuseColor,
    ECALL_setGlowColor,
    ECALL_setModel,
    ECALL_setCollisionShape,
    ECALL_setCollisionFlags,
    ECALL_setHealth,
    ECALL_setRoutine,
    ECALL_setOwner,
    ECALL_setVelocity,
    ECALL_setAccel,
    ECALL_setPower,
    ECALL_setPosition,
    ECALL_setScale,
    ECALL_setAxis,
    ECALL_setAxis1 = ECALL_setAxis,
    ECALL_setAxis2,
    ECALL_setRotate,
    ECALL_setRotate1 = ECALL_setRotate,
    ECALL_setRotate2,
    ECALL_setRotateSpeed,
    ECALL_setRotateSpeed1 = ECALL_setRotateSpeed,
    ECALL_setRotateSpeed2,
    ECALL_setDirection,
    ECALL_setSpeed,
    ECALL_setLightRadius,
    ECALL_setExplosionSE,
    ECALL_setExplosionChannel,

    ECALL_End,
};

enum ENTITY_QUERY
{
    EQUERY_getRefCount,
    EQUERY_getDiffuseColor,
    EQUERY_getGlowColor,
    EQUERY_getModel,
    EQUERY_getCollisionHandle,
    EQUERY_getCollisionFlags,
    EQUERY_getHealth,
    EQUERY_getOwner,
    EQUERY_getVelocity,
    EQUERY_getPower,
    EQUERY_getPosition,
    EQUERY_getScale,
    EQUERY_getAxis,
    EQUERY_getAxis1 = EQUERY_getAxis,
    EQUERY_getAxis2,
    EQUERY_getRotate,
    EQUERY_getRotate1 = EQUERY_getRotate,
    EQUERY_getRotate2,
    EQUERY_getDirection,
    EQUERY_getSpeed,
    EQUERY_getRotateSpeed,
    EQUERY_getRotateSpeed1 = ECALL_setRotateSpeed,
    EQUERY_getRotateSpeed2,

    EQUERY_End,
};


class IEntity;
template<class T>
inline T _atomicQuery(IEntity *e, ENTITY_QUERY qid)
{
    variant v;
    if(!e->query(qid, v)) {
        istPrint("query failed. entity: 0x%x query: %d\n", e->getHandle(), qid);
    }
    return v.cast<T>();
}

#define atomicCall(entity, funcname, ...) entity->call(ECALL_##funcname, __VA_ARGS__)
#define atomicQuery(entity, funcname, T) _atomicQuery<T>(entity, EQUERY_##funcname)

} // namespace atomic
#endif // __atomic_Game_EntityQuery__
