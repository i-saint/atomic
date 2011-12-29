#ifndef __atomic_Game_EntityQuery__
#define __atomic_Game_EntityQuery__


#define DEFINE_ECALL0(funcname)         \
    case ECALL_##funcname: funcname(); return true;

#define DEFINE_ECALL1(funcname, type)   \
    case ECALL_##funcname: funcname(v.cast<type>()); return true;

#define DEFINE_EQUERY(funcname)    \
    case EQUERY_##funcname: v=funcname(); return true;

#define atomicCall(entity, funcname, ...) entity->call(ECALL_##funcname, __VA_ARGS__)
#define atomicQuery(entity, funcname, ...) entity->query(EQUERY_##funcname, __VA_ARGS__)

namespace atomic {

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
    ECALL_setHealth,
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
    ECALL_addEventListener,

    ECALL_END,
};

enum ENTITY_QUERY
{
    EQUERY_getRefCount,
    EQUERY_getDiffuseColor,
    EQUERY_getGlowColor,
    EQUERY_getModel,
    EQUERY_getHealth,
    EQUERY_getPosition,
    EQUERY_getScale,
    EQUERY_getAxis,
    EQUERY_getAxis1 = EQUERY_getAxis,
    EQUERY_getAxis2,
    EQUERY_getRotate,
    EQUERY_getRotate1 = EQUERY_getRotate,
    EQUERY_getRotate2,
    EQUERY_getRotateSpeed,
    EQUERY_getRotateSpeed1 = ECALL_setRotateSpeed,
    EQUERY_getRotateSpeed2,

    EQUERY_END,
};

} // namespace atomic
#endif // __atomic_Game_EntityQuery__
