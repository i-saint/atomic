#include "stdafx.h"
#include "EntityUtil.h"
#include "Game/CollisionModule.h"
#include "Game/BulletModule.h"

namespace atm {


void UpdateCollisionSphere(CollisionSphere &o, const vec3& pos, float32 r)
{
    o.pos_r = vec4(pos, r);
    o.bb.ur = o.pos_r + vec4( r, r, r, 0.0f);
    o.bb.bl = o.pos_r + vec4(-r,-r,-r, 0.0f);
}

void UpdateCollisionBox(CollisionBox &o, const mat4& t, const vec3 &size)
{
    simdmat4 st(t);
    vec3 vertices[] = {
        vec3(glm::vec4_cast(st * simdvec4( size.x, size.y, size.z, 0.0f))),
        vec3(glm::vec4_cast(st * simdvec4(-size.x, size.y, size.z, 0.0f))),
        vec3(glm::vec4_cast(st * simdvec4(-size.x,-size.y, size.z, 0.0f))),
        vec3(glm::vec4_cast(st * simdvec4( size.x,-size.y, size.z, 0.0f))),
        vec3(glm::vec4_cast(st * simdvec4( size.x, size.y,-size.z, 0.0f))),
        vec3(glm::vec4_cast(st * simdvec4(-size.x, size.y,-size.z, 0.0f))),
        vec3(glm::vec4_cast(st * simdvec4(-size.x,-size.y,-size.z, 0.0f))),
        vec3(glm::vec4_cast(st * simdvec4( size.x,-size.y,-size.z, 0.0f))),
    };
    vec3 normals[6] = {
        glm::normalize(glm::cross(vertices[3]-vertices[0], vertices[4]-vertices[0])),
        glm::normalize(glm::cross(vertices[5]-vertices[1], vertices[2]-vertices[1])),
        glm::normalize(glm::cross(vertices[7]-vertices[3], vertices[2]-vertices[3])),
        glm::normalize(glm::cross(vertices[1]-vertices[0], vertices[4]-vertices[0])),
        glm::normalize(glm::cross(vertices[1]-vertices[0], vertices[3]-vertices[0])),
        glm::normalize(glm::cross(vertices[7]-vertices[4], vertices[5]-vertices[4])),
    };
    float32 distances[6] = {
        -glm::dot(vertices[0], normals[0]),
        -glm::dot(vertices[1], normals[1]),
        -glm::dot(vertices[0], normals[2]),
        -glm::dot(vertices[3], normals[3]),
        -glm::dot(vertices[0], normals[4]),
        -glm::dot(vertices[4], normals[5]),
    };

    const vec3 pos = vec3(t[3]);
    o.position = vec4(pos, 0.0f);
    o.trans = t;
    o.size = vec4(size, 0.0f);
    for(int32 i=0; i<_countof(normals); ++i) {
        o.planes[i] = vec4(normals[i], distances[i]);
    }
    o.bb.ur = o.bb.bl = vec4(vertices[0], 0.0f);
    for(int32 i=0; i<_countof(vertices); ++i) {
        vec4 t = vec4(vertices[i]+pos, 0.0f);
        o.bb.ur = glm::max(o.bb.ur, t);
        o.bb.bl = glm::min(o.bb.bl, t);
    }
}


vec3 GetNearestPlayerPosition(const vec3 &pos)
{
    vec3 ret;
    atmEnumlateEntity(
        [&](EntityHandle h){ return EntityGetClassID(h)==EC_Player; },
        [&](IEntity *e){ atmQuery(e, getPosition, ret); }
    );
    return ret;
}

void ShootSimpleBullet(EntityHandle owner, const vec3 &pos, const vec3 &vel)
{
    atmGetBulletModule()->shootBullet(pos, vel, owner);
}


IEntity* PutGroundBlock(IEntity *parent, CollisionGroup group, const vec3 &pos, const vec3 &size, const vec3 &dir, const vec3 &pivot)
{
    IEntity *e = atmCreateEntityT(GroundBlock);
    atmCall(e, setParent, parent ? parent->getHandle() : 0);
    atmCall(e, setPosition, pos);
    atmCall(e, setScale, size);
    atmCall(e, setDirection, dir);
    atmCall(e, setPivot, pivot);
    atmCall(e, setCollisionGroup, group);
    return e;
}
IEntity* PutGroundBlockByBox(IEntity *parent, CollisionGroup group, const vec3 &_box_min, const vec3 &_box_max, const vec3 &dir)
{
    vec3 box_min = glm::min(_box_min, _box_max);
    vec3 box_max = glm::max(_box_min, _box_max);
    vec3 size = box_max-box_min;
    vec3 half_size = size*0.5f;
    IEntity *e = atmCreateEntityT(GroundBlock);
    atmCall(e, setParent, parent ? parent->getHandle() : 0);
    atmCall(e, setPosition, box_min+half_size);
    atmCall(e, setScale, size);
    atmCall(e, setDirection, dir);
    atmCall(e, setCollisionGroup, group);
    return e;
}


IEntity* PutFluidFilter(IEntity *parent, CollisionGroup group, const vec3 &pos, const vec3 &size, const vec3 &dir, const vec3 &pivot)
{
    IEntity *e = atmCreateEntityT(FluidFilter);
    atmCall(e, setParent, parent ? parent->getHandle() : 0);
    atmCall(e, setPosition, pos);
    atmCall(e, setScale, size);
    atmCall(e, setDirection, dir);
    atmCall(e, setPivot, pivot);
    atmCall(e, setCollisionGroup, group);
    return e;
}

IEntity* PutFluidFilterByBox(IEntity *parent, CollisionGroup group, const vec3 &_box_min, const vec3 &_box_max, const vec3 &dir)
{
    vec3 box_min = glm::min(_box_min, _box_max);
    vec3 box_max = glm::max(_box_min, _box_max);
    vec3 size = box_max-box_min;
    vec3 half_size = size*0.5f;
    IEntity *e = atmCreateEntityT(FluidFilter);
    atmCall(e, setParent, parent ? parent->getHandle() : 0);
    atmCall(e, setPosition, box_min+half_size);
    atmCall(e, setScale, size);
    atmCall(e, setDirection, dir);
    atmCall(e, setCollisionGroup, group);
    return e;
}


IEntity* PutRigidFilter(IEntity *parent, CollisionGroup group, const vec3 &pos, const vec3 &size, const vec3 &dir, const vec3 &pivot)
{
    IEntity *e = atmCreateEntityT(RigidFilter);
    atmCall(e, setParent, parent ? parent->getHandle() : 0);
    atmCall(e, setPosition, pos);
    atmCall(e, setScale, size);
    atmCall(e, setDirection, dir);
    atmCall(e, setPivot, pivot);
    atmCall(e, setCollisionGroup, group);
    return e;
}
IEntity* PutRigidFilterByBox(IEntity *parent, CollisionGroup group, const vec3 &_box_min, const vec3 &_box_max, const vec3 &dir)
{
    vec3 box_min = glm::min(_box_min, _box_max);
    vec3 box_max = glm::max(_box_min, _box_max);
    vec3 size = box_max-box_min;
    vec3 half_size = size*0.5f;
    IEntity *e = atmCreateEntityT(RigidFilter);
    atmCall(e, setParent, parent ? parent->getHandle() : 0);
    atmCall(e, setPosition, box_min+half_size);
    atmCall(e, setScale, size);
    atmCall(e, setDirection, dir);
    atmCall(e, setCollisionGroup, group);
    return e;
}

} // namespace atm
