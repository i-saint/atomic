#include "stdafx.h"
#include "types.h"
#include "Game/AtomicApplication.h"
#include "Game/AtomicGame.h"
#include "Game/World.h"
#include "Game/Collision.h"
#include "Game/Entity.h"
#include "Game/EntityClass.h"
#include "Game/EntityQuery.h"
#include "Graphics/ResourceManager.h"
#include "Graphics/Shader.h"
#include "Util.h"

namespace atomic {

void FillScreen( const vec4 &color )
{
    AtomicShader *sh_fill   = atomicGetShader(SH_FILL);
    VertexArray *va_quad    = atomicGetVertexArray(VA_SCREEN_QUAD);
    Buffer *ubo_params      = atomicGetUniformBuffer(UBO_FILL_PARAMS);
    static uint32 location  = sh_fill->getUniformBlockIndex("fill_params");

    FillParams params;
    params.color = color;
    MapAndWrite(*ubo_params, &params, sizeof(params));

    sh_fill->bind();
    sh_fill->setUniformBlock(location, GLSL_FILL_BINDING, ubo_params->getHandle());
    va_quad->bind();
    glDrawArrays(GL_QUADS, 0, 4);
    sh_fill->unbind();
}


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


vec4 GetNearestPlayerPosition(const vec4 &pos)
{
    if(IEntity *player = atomicGetEntity( EntityCreateHandle(ECID_Player, ESID_Player, 0) )) {
        return atomicQuery(player, getPosition, vec4);
    }
    return vec4();
}

void ShootSimpleBullet(EntityHandle owner, const vec4 &pos, const vec4 &vel)
{
    IEntity *e = atomicCreateEntity(Bullet_Simple);
    atomicCall(e, setOwner, owner);
    atomicCall(e, setPosition, pos);
    atomicCall(e, setVelocity, vel);
}

void CreateDateString(char *buf, uint32 len)
{
    time_t t = ::time(0);
    tm *l = ::localtime(&t);
    sprintf_s(buf, len, "%d/%02d/%02d %02d:%02d:%02d",
        l->tm_year+1900, l->tm_mon+1, l->tm_mday, l->tm_hour, l->tm_min, l->tm_sec);
}


} // namespace atomic
