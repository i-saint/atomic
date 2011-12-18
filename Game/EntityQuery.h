#ifndef __atomic_Game_EntityQuery__
#define __atomic_Game_EntityQuery__


#define DEFINE_ECALL0(funcname)         \
    case ECALL_##funcname: funcname(); return true;

#define DEFINE_ECALL1(funcname, type)   \
    case ECALL_##funcname: funcname(v.cast<type>()); return true;

#define DEFINE_EQUERY(funcname)    \
    case EQUERY_##funcname: v=funcname(); return true;


namespace atomic {

enum ENTITY_CALL
{
    ECALL_kill,
    ECALL_destroy,
    ECALL_setRefCount,
    ECALL_addRef,
    ECALL_release,
    ECALL_setHealth,
    ECALL_setPosition,
    ECALL_setScale,
    ECALL_setAxis,
    ECALL_setAxis1 = ECALL_setAxis,
    ECALL_setAxis2,
    ECALL_setRotation,
    ECALL_setRotation1 = ECALL_setRotation,
    ECALL_setRotation2,
    ECALL_addEventListener,

    ECALL_END,
};

enum ENTITY_QUERY
{
    EQUERY_getRefCount,
    EQUERY_getHealth,
    EQUERY_getPosition,
    EQUERY_getScale,
    EQUERY_getAxis,
    EQUERY_getAxis1 = EQUERY_getAxis,
    EQUERY_getAxis2,
    EQUERY_getRotation,
    EQUERY_getRotation1 = EQUERY_getRotation,
    EQUERY_getRotation2,

    EQUERY_END,
};

} // namespace atomic
#endif // __atomic_Game_EntityQuery__
