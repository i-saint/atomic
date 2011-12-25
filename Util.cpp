#include "stdafx.h"
#include "types.h"
#include "Game/AtomicApplication.h"
#include "Game/AtomicGame.h"
#include "Game/World.h"
#include "GPGPU/SPH.cuh"
#include "Util.h"

namespace atomic {

vec4 GenRotateAxis()
{
    vec4 axis( atomicGenRandFloat(), atomicGenRandFloat(), atomicGenRandFloat(), 0.0f );
    axis -= vec4(0.5f, 0.5f, 0.5f, 0.0f);
    axis *= 2.0f;
    return glm::normalize(axis);
}

void CreateRigidSphere(sphRigidSphere &o, EntityHandle h, const vec4& pos, float32 r)
{
    o.owner_handle = h;
    o.pos_r = make_float4(pos.x, pos.y, pos.z, r);
    o.bb.ur = o.pos_r + make_float4( r, r, r, 0.0f);
    o.bb.bl = o.pos_r + make_float4(-r,-r,-r, 0.0f);
    //IST_PRINT("sphere bb max: %.2f, %.2f, %.2f\n", o.bb.ur.x, o.bb.ur.y, o.bb.ur.z);
    //IST_PRINT("sphere bb min: %.2f, %.2f, %.2f\n", o.bb.bl.x, o.bb.bl.y, o.bb.bl.z);
}

void CreateRigidBox(sphRigidBox &o, EntityHandle h, const mat4& t, const vec4 &size)
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

    o.owner_handle = h;
    o.position = (float4&)(pos);
    o.position.w = 0.0f;
    //o.position = make_float4(0.0f);
    for(int32 i=0; i<_countof(planes); ++i) {
        o.planes[i] = reinterpret_cast<float4&>(t * planes[i]);
        o.planes[i].w = distances[i];
    }
    o.bb.ur = (float4&)(pos + vec4( r, r, r, 0.0f));
    o.bb.bl = (float4&)(pos + vec4(-r,-r,-r, 0.0f));
    //IST_PRINT("box bb max: %.2f, %.2f, %.2f\n", o.bb.ur.x, o.bb.ur.y, o.bb.ur.z);
    //IST_PRINT("box bb min: %.2f, %.2f, %.2f\n", o.bb.bl.x, o.bb.bl.y, o.bb.bl.z);
}

} // namespace atomic
