#ifndef __atomic_Game_Character_Enemy__
#define __atomic_Game_Character_Enemy__

#include "Game/Entity.h"
#include "Game/EntityQuery.h"
#include "Attributes.h"

namespace atomic {

    enum ENEMY_ROUTINE_CLASS_ID
    {
        ENEMY_ROUTINE_HOMING_PLAYER,
        ENEMY_ROUTINE_END,
    };

    class IRoutine
    {
    public:
        virtual ~IRoutine() {}
        virtual void update(float32 dt)=0;
        virtual void updateAsync(float32 dt)=0;
        virtual void draw() {}

        virtual bool call(uint32 call_id, const variant &v) { return false; }
        virtual bool query(uint32 query_id, variant &v) const { return false; }
    };


    class Enemy_Cube;
    class Enemy_Sphere;

} // namespace atomic
#endif // __atomic_Game_Character_Enemy__
