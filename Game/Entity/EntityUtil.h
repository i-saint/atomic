#ifndef atm_Game_Entity_EntityUtil_h
#define atm_Game_Entity_EntityUtil_h

#include "Game/AtomicGame.h"
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

template<class T>
inline mat4 GetRotationMatrix(T *_this)
{
    mat4 rot;
    if(atmQuery(_this, computeRotationMatrix, rot)) {
        return rot;
    }
    return _this->getTransformMatrix();
}


template<class T> struct TypeS;
template<> struct TypeS<bool>   {static const char* get(){return "bool";}};
template<> struct TypeS<int32>  {static const char* get(){return "int32";}};
template<> struct TypeS<uint32> {static const char* get(){return "uint32";}};
template<> struct TypeS<float32>{static const char* get(){return "float32";}};
template<> struct TypeS<vec2>   {static const char* get(){return "vec2";}};
template<> struct TypeS<vec3>   {static const char* get(){return "vec3";}};
template<> struct TypeS<vec4>   {static const char* get(){return "vec4";}};

inline void Jsonize(stl::string &out, const char *name, const char *getter, const char *setter, bool v) {
    out += ist::Format("{\"name\":\"%s\",\"type\":\"bool\",\"getter\":\"%s\",\"setter\":\"%s\",\"value\":%d},",
        name+2, getter, setter, (int32)v);
}
inline void Jsonize(stl::string &out, const char *name, const char *getter, const char *setter, int32 v) {
    out += ist::Format("{\"name\":\"%s\",\"type\":\"int32\",\"getter\":\"%s\",\"setter\":\"%s\",\"value\":%d},",
        name+2, getter, setter, v);
}
inline void Jsonize(stl::string &out, const char *name, const char *getter, const char *setter, uint32 v) {
    out += ist::Format("{\"name\":\"%s\",\"type\":\"uint32\",\"getter\":\"%s\",\"setter\":\"%s\",\"value\":%u},",
        name+2, getter, setter, v);
}
inline void Jsonize(stl::string &out, const char *name, const char *getter, const char *setter, float32 v) {
    out += ist::Format("{\"name\":\"%s\",\"type\":\"float32\",\"getter\":\"%s\",\"setter\":\"%s\",\"value\":%.2f},",
        name+2, getter, setter, v);
}
inline void Jsonize(stl::string &out, const char *name, const char *getter, const char *setter, const vec2 &v) {
    out += ist::Format("{\"name\":\"%s\",\"type\":\"vec2\",\"getter\":\"%s\",\"setter\":\"%s\",\"value\":[%.2f,%.2f]},",
        name+2, getter, setter, v.x,v.y);
}
inline void Jsonize(stl::string &out, const char *name, const char *getter, const char *setter, const vec3 &v) {
    out += ist::Format("{\"name\":\"%s\",\"type\":\"vec3\",\"getter\":\"%s\",\"setter\":\"%s\",\"value\":[%.2f,%.2f,%.2f]},",
        name+2, getter, setter, v.x,v.y,v.z);
}
inline void Jsonize(stl::string &out, const char *name, const char *getter, const char *setter, const vec4 &v) {
    out += ist::Format("{\"name\":\"%s\",\"type\":\"vec4\",\"getter\":\"%s\",\"setter\":\"%s\",\"value\":[%.2f,%.2f,%.2f,%.2f]},",
        name+2, getter, setter, v.x,v.y,v.z,v.w);
}
#define atmJsonizeMember(Out,V,Getter,Setter) Jsonize(Out, #V, #Getter, #Setter, V)
#define atmJsonizeMemberFunction(Out,F)

} // namespace atm
#endif // atm_Game_Entity_EntityUtil_h
