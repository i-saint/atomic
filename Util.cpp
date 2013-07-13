#include "stdafx.h"
#include <ctime>
#include "types.h"
#include "Game/AtomicApplication.h"
#include "Game/AtomicGame.h"
#include "Game/World.h"
#include "Game/CollisionModule.h"
#include "Game/EntityModule.h"
#include "Game/EntityClass.h"
#include "Game/EntityQuery.h"
#include "Graphics/ResourceManager.h"
#include "Graphics/Renderer.h"
#include "Graphics/Shader.h"
#include "Util.h"

namespace atm {

void FillScreen( const vec4 &color )
{
    i3d::DeviceContext *dc = atmGetGLDeviceContext();
    AtomicShader *sh_fill   = atmGetShader(SH_FILL);
    VertexArray *va_quad    = atmGetVertexArray(VA_SCREEN_QUAD);
    Buffer *ubo_params      = atmGetUniformBuffer(UBO_FILL_PARAMS);
    static uint32 location  = sh_fill->getUniformBlockIndex("fill_params");

    FillParams params;
    params.Color = color;
    MapAndWrite(dc, ubo_params, &params, sizeof(params));

    sh_fill->bind();
    sh_fill->setUniformBlock(location, GLSL_FILL_BINDING, ubo_params);
    dc->setVertexArray(va_quad);
    dc->draw(i3d::I3D_QUADS, 0, 4);
    sh_fill->unbind();
}


float32 Interpolate(const ControlPoint &v1, const ControlPoint &v2, float32 time)
{
    float32 u = (time - v1.time) / (v2.time-v1.time);
    float32 r;
    switch(v1.interp) {
    case atmE_None:   r=v1.value; break;
    case atmE_Linear: r=ist::interpolate_lerp(v1.value, v2.value, u); break;
    case atmE_Decel:  r=ist::interpolate_sin90(v1.value, v2.value, u); break;
    case atmE_Accel:  r=ist::interpolate_cos90inv(v1.value, v2.value, u); break;
    case atmE_Smooth: r=ist::interpolate_cos180inv(v1.value, v2.value, u); break;
    case atmE_Bezier: r=ist::interpolate_bezier(v1.value, v1.bez_out, v2.bez_in, v2.value, u); break;
    }
    return r;
}


vec2 GenRandomVector2()
{
    vec2 axis( atmGenRandFloat(), atmGenRandFloat() );
    axis -= vec2(0.5f, 0.5f);
    axis *= 2.0f;
    return axis;
}

vec3 GenRandomVector3()
{
    vec3 axis( atmGenRandFloat(), atmGenRandFloat(), atmGenRandFloat() );
    axis -= 0.5f;
    axis *= 2.0f;
    return axis;
}

vec2 GenRandomUnitVector2()
{
    return glm::normalize(GenRandomVector2());
}

vec3 GenRandomUnitVector3()
{
    return glm::normalize(GenRandomVector3());
}

void CreateDateString(char *buf, uint32 len)
{
    time_t t = ::time(0);
    tm *l = ::localtime(&t);
    istSNPrintf(buf, len, "%d/%02d/%02d %02d:%02d:%02d",
        l->tm_year+1900, l->tm_mon+1, l->tm_mday, l->tm_hour, l->tm_min, l->tm_sec);
}


} // namespace atm
