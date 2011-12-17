#ifndef __atomic_Game_EntityQuery__
#define __atomic_Game_EntityQuery__

namespace atomic {

enum ENTITY_CALL
{
    ECALL_setHealth,
    ECALL_setPosition,
    ECALL_addEventListener,
    ECALL_END,
};

enum ENTITY_QUERY
{
    EQUERY_getHealth,
    EQUERY_getPosition,
    EQUERY_END,
};

} // namespace atomic
#endif // __atomic_Game_EntityQuery__
