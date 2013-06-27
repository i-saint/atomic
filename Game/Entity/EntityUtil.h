#ifndef atm_Game_Entity_EntityUtil_h
#define atm_Game_Entity_EntityUtil_h

#include "Game/AtomicGame.h"
#include "Game/CollisionModule.h"
#include "Game/World.h"
#include "Game/EntityModule.h"
#include "Game/EntityQuery.h"

namespace atm {

void UpdateCollisionSphere(CollisionSphere &o, const vec3& pos, float32 r);
void UpdateCollisionBox(CollisionBox &o, const mat4& t, const vec3 &size);

vec3 GetNearestPlayerPosition(const vec3 &pos);
void ShootSimpleBullet(EntityHandle owner, const vec3 &pos, const vec3 &vel);


inline size_t SweepDeadEntities(stl::vector<EntityHandle> &cont)
{
    size_t ret = 0;
    for(size_t i=0; i<cont.size(); ++i) {
        EntityHandle v = cont[i];
        if(v) {
            if(atmGetEntity(v)==nullptr) {
                cont[i] = 0;
            }
            else {
                ++ret;
            }
        }
    }
    return ret;
}

template<size_t L>
inline size_t SweepDeadEntities(EntityHandle (&cont)[L])
{
    size_t ret = 0;
    for(size_t i=0; i<L; ++i) {
        EntityHandle v = cont[i];
        if(v) {
            if(atmGetEntity(v)==nullptr) {
                cont[i] = 0;
            }
            else {
                ++ret;
            }
        }
    }
    return ret;
}

inline size_t SweepDeadEntities(EntityHandle &v)
{
    if(v) {
        if(atmGetEntity(v)==nullptr) {
            v = 0;
        }
        else {
            return 1;
        }
    }
    return 0;
}

template<class F>
inline void EachEntities(stl::vector<EntityHandle> &cont, const F &f)
{
    for(size_t i=0; i<cont.size(); ++i) {
        if(cont[i]) { f(cont[i]); }
    }
}

} // namespace atm
#endif // atm_Game_Entity_EntityUtil_h
