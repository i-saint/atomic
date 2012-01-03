#include "stdafx.h"
#include "types.h"
#include "Game/AtomicApplication.h"
#include "Game/AtomicGame.h"
#include "Game/World.h"
#include "AtomicRenderingSystem.h"
#include "Renderer.h"
#include "Util.h"

namespace atomic {



PassDeferredShading_DirectionalLights::PassDeferredShading_DirectionalLights()
{
    m_shader        = atomicGetShader(SH_DIRECTIONALLIGHT);
    m_va_quad       = atomicGetVertexArray(VA_SCREEN_QUAD);
    m_vbo_instance  = atomicGetVertexBuffer(VBO_DIRECTIONALLIGHT_INSTANCES);
    m_instances.reserve(ATOMIC_MAX_DIRECTIONAL_LIGHTS);
}

void PassDeferredShading_DirectionalLights::beforeDraw()
{
    m_instances.clear();
}

void PassDeferredShading_DirectionalLights::draw()
{
    const uint32 num_instances = m_instances.size();
    MapAndWrite(*m_vbo_instance, &m_instances[0], sizeof(light_t)*num_instances);

    const VertexDescriptor descs[] = {
        {GLSL_INSTANCE_DIRECTION,I3D_FLOAT,4,  0, false, 1},
        {GLSL_INSTANCE_COLOR,    I3D_FLOAT,4, 16, false, 1},
        {GLSL_INSTANCE_AMBIENT,  I3D_FLOAT,4, 32, false, 1},
    };
    m_shader->bind();
    m_va_quad->bind();
    m_va_quad->setAttributes(*m_vbo_instance, sizeof(DirectionalLight), descs, _countof(descs));
    glDrawArraysInstanced(GL_QUADS, 0, 4, num_instances);
    m_va_quad->unbind();
    m_shader->unbind();
}

void PassDeferredShading_DirectionalLights::addInstance( const DirectionalLight& v )
{
    m_instances.push_back(v);
}



PassDeferredShading_PointLights::PassDeferredShading_PointLights()
{
    m_shader        = atomicGetShader(SH_POINTLIGHT);
    m_ibo_sphere    = atomicGetIndexBuffer(IBO_SPHERE);
    m_va_sphere     = atomicGetVertexArray(VA_UNIT_SPHERE);
    m_vbo_instance  = atomicGetVertexBuffer(VBO_POINTLIGHT_INSTANCES);
    m_instances.reserve(1024);
}

void PassDeferredShading_PointLights::beforeDraw()
{
    m_instances.clear();
}

void PassDeferredShading_PointLights::draw()
{
    const uint32 num_instances = m_instances.size();
    MapAndWrite(*m_vbo_instance, &m_instances[0], sizeof(PointLight)*num_instances);

    const VertexDescriptor descs[] = {
        {GLSL_INSTANCE_POSITION,I3D_FLOAT,4, 0, false, 1},
        {GLSL_INSTANCE_COLOR,   I3D_FLOAT,4,16, false, 1},
        {GLSL_INSTANCE_PARAM,   I3D_FLOAT,4,32, false, 1},
    };

    m_shader->bind();
    m_va_sphere->bind();
    m_va_sphere->setAttributes(*m_vbo_instance, sizeof(PointLight), descs, _countof(descs));
    m_ibo_sphere->bind();
    glDrawElementsInstanced(GL_QUADS, (16-1)*(32)*4, GL_UNSIGNED_INT, 0, num_instances);
    m_ibo_sphere->unbind();
    m_va_sphere->unbind();
    m_shader->unbind();
}

} // namespace atomic
