#ifndef __atomic_Game_Character_Enemy__
#define __atomic_Game_Character_Enemy__

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
        virtual void update(float32 dt);
        virtual void updateAsync(float32 dt);
    };


    class Enemy_Cube;
    class Enemy_Sphere;

} // namespace atomic
#endif // __atomic_Game_Character_Enemy__
