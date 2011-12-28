#include "stdafx.h"
#include "types.h"
#include "Game/AtomicApplication.h"
#include "Game/AtomicGame.h"
#include "Game/World.h"
#include "Game/Collision.h"
#include "GPGPU/SPH.cuh"
#include "Util.h"

namespace atomic {

vec4 GenRandomVector2()
{
    vec4 axis( atomicGenRandFloat(), atomicGenRandFloat(), 0.0f, 0.0f );
    axis -= vec4(0.5f, 0.5f, 0.0f, 0.0f);
    axis *= 2.0f;
    return axis;
}

vec4 GenRandomVector3()
{
    vec4 axis( atomicGenRandFloat(), atomicGenRandFloat(), atomicGenRandFloat(), 0.0f );
    axis -= vec4(0.5f, 0.5f, 0.5f, 0.0f);
    axis *= 2.0f;
    return axis;
}

vec4 GenRandomUnitVector2()
{
    return glm::normalize(GenRandomVector2());
}

vec4 GenRandomUnitVector3()
{
    return glm::normalize(GenRandomVector3());
}

void UpdateCollisionSphere(CollisionSphere &o, const vec4& pos, float32 r)
{
    o.pos_r = vec4(pos.x, pos.y, pos.z, r);
    o.bb.ur = o.pos_r + vec4( r, r, r, 0.0f);
    o.bb.bl = o.pos_r + vec4(-r,-r,-r, 0.0f);
}

void UpdateCollisionBox(CollisionBox &o, const mat4& t, const vec4 &size)
{
    const vec4 pos = t*vec4(0.0f,0.0f,0.0f,1.0f);
    const float r = glm::length(size);
    const vec4 planes[6] = {
        vec4( 1.0f, 0.0f, 0.0f, 0.0f),
        vec4(-1.0f, 0.0f, 0.0f, 0.0f),
        vec4( 0.0f, 1.0f, 0.0f, 0.0f),
        vec4( 0.0f,-1.0f, 0.0f, 0.0f),
        vec4( 0.0f, 0.0f, 1.0f, 0.0f),
        vec4( 0.0f, 0.0f,-1.0f, 0.0f),
    };
    const float distances[6] = {
        -size.x,
        -size.x,
        -size.y,
        -size.y,
        -size.z,
        -size.z,
    };

    o.position = pos;
    o.position.w = 0.0f;
    //o.position = make_float4(0.0f);
    for(int32 i=0; i<_countof(planes); ++i) {
        o.planes[i] = t * planes[i];
        o.planes[i].w = distances[i];
    }
    o.bb.ur = pos + vec4( r, r, r, 0.0f);
    o.bb.bl = pos + vec4(-r,-r,-r, 0.0f);
}

} // namespace atomic
