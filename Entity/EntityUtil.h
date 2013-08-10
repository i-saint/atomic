#ifndef atm_Game_Entity_EntityUtil_h
#define atm_Game_Entity_EntityUtil_h

#include "Game/AtomicGame.h"
#include "Game/World.h"
#include "Game/EntityModule.h"
#include "Game/EntityQuery.h"

namespace atm {

struct CollisionSphere;
struct CollisionBox;
void UpdateCollisionSphere(CollisionSphere &o, const vec3& pos, float32 r);
void UpdateCollisionBox(CollisionBox &o, const mat4& t, const vec3 &size);

vec3 GetNearestPlayerPosition(const vec3 &pos);
void ShootSimpleBullet(EntityHandle owner, const vec3 &pos, const vec3 &vel);

IEntity* PutGroundBlock(IEntity *parent, CollisionGroup group, const vec3 &pos, const vec3 &size, const vec3 &dir=vec3(1.0f,0.0f,0.0f), const vec3 &pivot=vec3(0.0f,0.0f,-0.4f));
IEntity* PutGroundBlockByBox(IEntity *parent, CollisionGroup group, const vec3 &box_min, const vec3 &box_max, const vec3 &dir=vec3(1.0f,0.0f,0.0f));

IEntity* PutFluidFilter(IEntity *parent, const vec3 &pos, const vec3 &size, const vec3 &dir=vec3(1.0f,0.0f,0.0f), const vec3 &pivot=vec3(0.0f,0.0f,-0.4f));
IEntity* PutFluidFilterByBox(IEntity *parent, const vec3 &box_min, const vec3 &box_max, const vec3 &dir=vec3(1.0f,0.0f,0.0f));

IEntity* PutRigidFilter(IEntity *parent, CollisionGroup group, const vec3 &pos, const vec3 &size, const vec3 &dir=vec3(1.0f,0.0f,0.0f), const vec3 &pivot=vec3(0.0f,0.0f,-0.4f));
IEntity* PutRigidFilterByBox(IEntity *parent, CollisionGroup group, const vec3 &box_min, const vec3 &box_max, const vec3 &dir=vec3(1.0f,0.0f,0.0f));


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
template<> struct TypeS<ControlPoint> {static const char* get(){return "controlpoint";}};

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

inline void jsonize(stl::string &out, const ControlPoint &cp)
{
    out+=ist::Format("[%.2f,%.2f,%.2f,%.2f,%d]", cp.time, cp.value, cp.in, cp.out, cp.interp);
}

inline void jsonizeCurve1D(stl::string &out, const char *name, const char *adder, const char *clearer, const ControlPoints &cv)
{
    out += ist::Format("{\"name\":\"%s\",\"type\":\"curve1D\",\"add\":\"%s\",\"clear\":\"%s\",\"value\":[",
        name+2, adder, clearer);
    bool first = true;
    each(cv,[&](const ControlPoint &cp){
        if(!first){out+=",";} first=false;
        out+=ist::Format("[%.2f,%.2f,%.2f,%.2f,%d]", cp.time, cp.value, cp.in, cp.out, cp.interp);
    });
    out+="]},";
}

inline void jsonizeCurve2D(
    stl::string &out, const char *name,
    const char *addx, const char *clearx, const ControlPoints &cvx,
    const char *addy, const char *cleary, const ControlPoints &cvy )
{
    out += ist::Format("{\"name\":\"%s\",\"type\":\"curve2D\",\"add\":[\"%s\",\"%s\"],\"clear\":[\"%s\",\"%s\"],",
        name, addx, addy, clearx, cleary);
    out += "\"value\":{\"x\":[";
    {
        bool first = true;
        each(cvx,[&](const ControlPoint &cp){
            if(!first){out+=",";} first=false;
            out+=ist::Format("[%.2f,%.2f,%.2f,%.2f,%d]", cp.time, cp.value, cp.in, cp.out, cp.interp);
        });
    }
    out += "],\"y\":[";
    {
        bool first = true;
        each(cvy,[&](const ControlPoint &cp){
            if(!first){out+=",";} first=false;
            out+=ist::Format("[%.2f,%.2f,%.2f,%.2f,%d]", cp.time, cp.value, cp.in, cp.out, cp.interp);
        });
    }
    out+="]}},";
}


template<class R, class C>
inline void JsonizeMF(stl::string &out, const char *name, R (C::*f)()) {
    out += ist::Format("{\"name\":\"%s\",\"type\":\"function\",\"args\":[]},"
        , name);
}
template<class R, class C, class R0>
inline void JsonizeMF(stl::string &out, const char *name, R (C::*f)(R0)) {
    typedef typename std::remove_const<typename std::remove_reference<R0>::type>::type R0T;
    out += ist::Format("{\"name\":\"%s\",\"type\":\"function\",\"args\":[\"%s\"]},"
        , name, TypeS<R0T>::get());
}
template<class R, class C, class R0, class R1>
inline void JsonizeMF(stl::string &out, const char *name, R (C::*f)(R0,R1)) {
    typedef typename std::remove_const<typename std::remove_reference<R0>::type>::type R0T;
    typedef typename std::remove_const<typename std::remove_reference<R1>::type>::type R1T;
    out += ist::Format("{\"name\":\"%s\",\"type\":\"function\",\"args\":[\"%s\",\"%s\"]},"
        , name, TypeS<R0T>::get(), TypeS<R1T>::get());
}

#define atmJsonizeBlock(...) \
    void jsonize(stl::string &out) {\
        typedef std::remove_reference<decltype(*this)>::type this_t;\
        __VA_ARGS__\
    }

#define atmJsonizeSuper(T) T::jsonize(out);
#define atmJsonizeMember(V,Getter,Setter) Jsonize(out, #V, #Getter, #Setter, V);
#define atmJsonizeCurve1D(V,Add,Clear) jsonizeCurve1D(out, #V, #Add, #Clear, V);
#define atmJsonizeCurve2D(Name, AddX,ClearX,VX, AddY,ClearY,VY) jsonizeCurve2D(out, #Name, #AddX,#ClearX,VX, #AddY,#ClearY,VY);
#define atmJsonizeCall(F) JsonizeMF(out, #F, &this_t::F);

} // namespace atm
#endif // atm_Game_Entity_EntityUtil_h
