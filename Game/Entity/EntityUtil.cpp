#include "stdafx.h"
#include "EntityUtil.h"

namespace atm {


void UpdateCollisionSphere(CollisionSphere &o, const vec3& pos, float32 r)
{
    o.pos_r = vec4(pos, r);
    o.bb.ur = o.pos_r + vec4( r, r, r, 0.0f);
    o.bb.bl = o.pos_r + vec4(-r,-r,-r, 0.0f);
}

void UpdateCollisionBox(CollisionBox &o, const mat4& t, const vec3 &size)
{
    vec3 vertices[] = {
        vec3(t * vec4( size.x, size.y, size.z, 0.0f)),
        vec3(t * vec4(-size.x, size.y, size.z, 0.0f)),
        vec3(t * vec4(-size.x,-size.y, size.z, 0.0f)),
        vec3(t * vec4( size.x,-size.y, size.z, 0.0f)),
        vec3(t * vec4( size.x, size.y,-size.z, 0.0f)),
        vec3(t * vec4(-size.x, size.y,-size.z, 0.0f)),
        vec3(t * vec4(-size.x,-size.y,-size.z, 0.0f)),
        vec3(t * vec4( size.x,-size.y,-size.z, 0.0f)),
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
    vec3 ret = pos;
    atmEnumlateEntity(
        [&](EntityHandle h){ return EntityGetClassID(h)==EC_Player; },
        [&](IEntity *e){ atmQuery(e, getPosition, ret); }
    );
    return ret;
}

void ShootSimpleBullet(EntityHandle owner, const vec3 &pos, const vec3 &vel)
{
    IEntity *e = atmCreateEntity(Bullet_Simple);
    atmCall(e, setOwner, owner);
    atmCall(e, setPosition, pos);
    atmCall(e, setVelocity, vel);
}

} // namespace atm
