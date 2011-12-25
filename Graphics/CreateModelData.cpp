#include "stdafx.h"
#include "ist/ist.h"
#include "types.h"
#include "Graphics/ResourceID.h"
#include "Graphics/ParticleSet.h"
#include "Graphics/CreateModelData.h"
#include "GPGPU/SPH.cuh"
#include "shader/Semantics.glslh"
#include <math.h>


namespace atomic {


void CreateSphere(
    VertexArray& va, VertexBufferObject& vbo, IndexBufferObject& ibo,
    float32 radius, uint32 div_xz, uint32 div_y)
{
    struct __declspec(align(16)) vertex_t
    {
        vec4 pos;
        vec4 normal;
    };
    const uint32 vertex_size = sizeof(vertex_t)*div_y*div_xz;
    const uint32 index_size = sizeof(uint32)*(div_y-1)*(div_xz)*4;
    vertex_t *vert = (vertex_t*)IST_ALIGNED_MALLOC(vertex_size, 16);
    uint32 *index = (uint32*)IST_ALIGNED_MALLOC(index_size, 16);

    for(uint32 yi=0; yi<div_y; ++yi) {
        float ang = glm::radians(180.0f/(div_y-1)*yi-90.0f);
        vert[div_xz*yi].pos = vec4(cos(ang)*radius, sin(ang)*radius, 0, 1.0);
    }

    mat4 rot;
    for(uint32 xzi=0; xzi<div_xz; ++xzi) {
        for(uint32 yi=0; yi<div_y; ++yi) {
            vertex_t* vp = &vert[div_xz*yi];
            vec4 rotated = rot * vp[0].pos;
            vp[xzi].pos = rotated;
            vp[xzi].normal = glm::normalize(vec4(rotated.x, rotated.y, rotated.z, 0.0f));
        }
        rot = glm::rotate(mat4(), 360.0f/div_xz*xzi, vec3(0.0f, 1.0f, 0.0f));
    }

    uint32 *ci = &index[0];
    for(uint32 yi=0; yi<div_y-1; ++yi) {
        for(uint32 xzi=0; xzi<div_xz; ++xzi) {
            ci[0] = div_xz*(yi)  + xzi;
            ci[1] = div_xz*(yi)  + ((xzi+1)%div_xz);
            ci[2] = div_xz*(yi+1)+ ((xzi+1)%div_xz);
            ci[3] = div_xz*(yi+1)+ xzi;
            ci+=4;
        }
    }

    VertexArray::Descriptor descs[] = {
        {GLSL_POSITION, VertexArray::TYPE_FLOAT,4,  0, false, 0},
        {GLSL_NORMAL,   VertexArray::TYPE_FLOAT,4, 16, false, 0},
    };
    vbo.allocate(vertex_size, VertexBufferObject::USAGE_STATIC, vert);
    ibo.allocate(index_size, IndexBufferObject::USAGE_STATIC, index);
    va.setAttributes(vbo, sizeof(vertex_t), descs, _countof(descs));

    IST_FREE(index);
    IST_FREE(vert);
}

void CreateScreenQuad( VertexArray& va, VertexBufferObject& vbo )
{
    struct __declspec(align(16)) vertex_t
    {
        vec2 pos;
        vec2 tex;
    } v[4] = {
        {vec2( 1.0f, 1.0f), vec2(1.0, 1.0)},
        {vec2(-1.0f, 1.0f), vec2(0.0, 1.0)},
        {vec2(-1.0f,-1.0f), vec2(0.0, 0.0)},
        {vec2( 1.0f,-1.0f), vec2(1.0, 0.0)},
    };
    vbo.allocate(sizeof(v), VertexBufferObject::USAGE_STATIC, v);

    VertexArray::Descriptor descs[] = {
        {GLSL_POSITION,  VertexArray::TYPE_FLOAT,2, 0, false, 0},
        {GLSL_TEXCOORD0, VertexArray::TYPE_FLOAT,2, 8, false, 0},
    };
    va.setAttributes(vbo, sizeof(vertex_t), descs, _countof(descs));
}

void CreateBloomLuminanceQuads( VertexArray& va, VertexBufferObject& vbo )
{
    struct __declspec(align(16)) vertex_t
    {
        vec2 pos;
        vec2 tex;
    } v[16] = {
        {vec2( 0.0, 1.0), vec2(1.0, 1.0)},
        {vec2(-1.0, 1.0), vec2(0.0, 1.0)},
        {vec2(-1.0,-1.0), vec2(0.0, 0.0)},
        {vec2( 0.0,-1.0), vec2(1.0, 0.0)},

        {vec2(0.5, 0.0), vec2(1.0, 1.0)},
        {vec2(0.0, 0.0), vec2(0.0, 1.0)},
        {vec2(0.0,-1.0), vec2(0.0, 0.0)},
        {vec2(0.5,-1.0), vec2(1.0, 0.0)},

        {vec2(0.75,-0.5), vec2(1.0, 1.0)},
        {vec2(0.5, -0.5), vec2(0.0, 1.0)},
        {vec2(0.5, -1.0), vec2(0.0, 0.0)},
        {vec2(0.75,-1.0), vec2(1.0, 0.0)},

        {vec2(0.875,-0.75), vec2(1.0, 1.0)},
        {vec2(0.75, -0.75), vec2(0.0, 1.0)},
        {vec2(0.75, -1.0 ), vec2(0.0, 0.0)},
        {vec2(0.875,-1.0 ), vec2(1.0, 0.0)},
    };
    vbo.allocate(sizeof(v), VertexBufferObject::USAGE_STATIC, v);

    VertexArray::Descriptor descs[] = {
        {GLSL_POSITION,  VertexArray::TYPE_FLOAT,2, 0, false, 0},
        {GLSL_TEXCOORD0, VertexArray::TYPE_FLOAT,2, 8, false, 0},
    };
    va.setAttributes(vbo, sizeof(vertex_t), descs, _countof(descs));
}

void CreateBloomBlurQuads( VertexArray& va, VertexBufferObject& vbo )
{
    const vec2 tp[4] = {vec2(0.0, 0.0), vec2(0.5,  0.0), vec2(0.75,  0.0 ), vec2(0.875,  0.0)};
    const vec2 ts[4] = {vec2(0.5, 1.0), vec2(0.25, 0.5), vec2(0.125, 0.25), vec2(0.0625, 0.125)};

    struct __declspec(align(16)) vertex_t
    {
        vec2 pos;
        vec2 tex;
        vec2 texmin;
        vec2 texmax;
    } v[16] = {
        {vec2( 0.0, 1.0), tp[0]+ts[0],             tp[0], tp[0]+ts[0]},
        {vec2(-1.0, 1.0), tp[0]+vec2(0.0,ts[0].y), tp[0], tp[0]+ts[0]},
        {vec2(-1.0,-1.0), tp[0],                   tp[0], tp[0]+ts[0]},
        {vec2( 0.0,-1.0), tp[0]+vec2(ts[0].x,0.0), tp[0], tp[0]+ts[0]},

        {vec2(0.5, 0.0), tp[1]+ts[1],              tp[1], tp[1]+ts[1]},
        {vec2(0.0, 0.0), tp[1]+vec2(0.0,ts[1].y),  tp[1], tp[1]+ts[1]},
        {vec2(0.0,-1.0), tp[1],                    tp[1], tp[1]+ts[1]},
        {vec2(0.5,-1.0), tp[1]+vec2(ts[1].x,0.0),  tp[1], tp[1]+ts[1]},

        {vec2(0.75,-0.5), tp[2]+ts[2],             tp[2], tp[2]+ts[2]},
        {vec2(0.5, -0.5), tp[2]+vec2(0.0,ts[2].y), tp[2], tp[2]+ts[2]},
        {vec2(0.5, -1.0), tp[2],                   tp[2], tp[2]+ts[2]},
        {vec2(0.75,-1.0), tp[2]+vec2(ts[2].x,0.0), tp[2], tp[2]+ts[2]},

        {vec2(0.875,-0.75), tp[3]+ts[3],            tp[3], tp[3]+ts[3]},
        {vec2(0.75, -0.75), tp[3]+vec2(0.0,ts[3].y),tp[3], tp[3]+ts[3]},
        {vec2(0.75, -1.0 ), tp[3],                  tp[3], tp[3]+ts[3]},
        {vec2(0.875,-1.0 ), tp[3]+vec2(ts[3].x,0.0),tp[3], tp[3]+ts[3]},
    };

    VertexArray::Descriptor descs[] = {
        {GLSL_POSITION,  VertexArray::TYPE_FLOAT,2, 0, false, 0},
        {GLSL_TEXCOORD0, VertexArray::TYPE_FLOAT,2, 8, false, 0},
        {GLSL_TEXCOORD1, VertexArray::TYPE_FLOAT,2,16, false, 0},
        {GLSL_TEXCOORD2, VertexArray::TYPE_FLOAT,2,24, false, 0},
    };

    vbo.allocate(sizeof(v), VertexBufferObject::USAGE_STATIC, v);
    va.setAttributes(vbo, sizeof(vertex_t), descs, _countof(descs));
}

void CreateBloomCompositeQuad( VertexArray& va, VertexBufferObject& vbo )
{
    const vec2 tp[4] = {vec2(0.0, 0.0), vec2(0.5,  0.0), vec2(0.75,  0.0 ), vec2(0.875,  0.0)};
    const vec2 ts[4] = {vec2(0.5, 1.0), vec2(0.25, 0.5), vec2(0.125, 0.25), vec2(0.0625, 0.125)};

    struct __declspec(align(16)) vertex_t
    {
        vec2 pos;
        vec2 tex[4];
    } v[4] = {
        {vec2( 1.0, 1.0), {tp[0]+ts[0], tp[1]+ts[1], tp[2]+ts[2], tp[3]+ts[3]}},
        {vec2(-1.0, 1.0), {tp[0]+vec2(0.0,ts[0].y), tp[1]+vec2(0.0,ts[1].y), tp[2]+vec2(0.0,ts[2].y), tp[3]+vec2(0.0,ts[3].y)}},
        {vec2(-1.0,-1.0), {tp[0], tp[1], tp[2], tp[3]}},
        {vec2( 1.0,-1.0), {tp[0]+vec2(ts[0].x,0.0), tp[1]+vec2(ts[1].x,0.0), tp[2]+vec2(ts[2].x,0.0), tp[3]+vec2(ts[3].x,0.0)}},
    };

    VertexArray::Descriptor descs[] = {
        {GLSL_POSITION,  VertexArray::TYPE_FLOAT,2, 0, false, 0},
        {GLSL_TEXCOORD0, VertexArray::TYPE_FLOAT,2, 8, false, 0},
        {GLSL_TEXCOORD1, VertexArray::TYPE_FLOAT,2,16, false, 0},
        {GLSL_TEXCOORD2, VertexArray::TYPE_FLOAT,2,24, false, 0},
        {GLSL_TEXCOORD3, VertexArray::TYPE_FLOAT,2,32, false, 0},
    };

    vbo.allocate(sizeof(v), VertexBufferObject::USAGE_STATIC, v);
    va.setAttributes(vbo, sizeof(vertex_t), descs, _countof(descs));
}

void CreateCube( VertexArray& va, VertexBufferObject& vbo, float32 len )
{
    const float32 half_len = len/2.0f;
    const vec3 ur = vec3( half_len, half_len, half_len);
    const vec3 bl = vec3(-half_len,-half_len,-half_len);
    const vec4 n[6] = {
        vec4( 1.0f, 0.0f, 0.0f, 0.0f),
        vec4(-1.0f, 0.0f, 0.0f, 0.0f),
        vec4( 0.0f, 1.0f, 0.0f, 0.0f),
        vec4( 0.0f,-1.0f, 0.0f, 0.0f),
        vec4( 0.0f, 0.0f, 1.0f, 0.0f),
        vec4( 0.0f, 0.0f,-1.0f, 0.0f),
    };

    struct __declspec(align(16)) vertex_t
    {
        vec4 pos;
        vec4 normal;
    } v[24] = {
        {vec4(ur[0], ur[1], ur[2], 1.0f), n[0]},
        {vec4(ur[0], bl[1], ur[2], 1.0f), n[0]},
        {vec4(ur[0], bl[1], bl[2], 1.0f), n[0]},
        {vec4(ur[0], ur[1], bl[2], 1.0f), n[0]},

        {vec4(bl[0], ur[1], ur[2], 1.0f), n[1]},
        {vec4(bl[0], ur[1], bl[2], 1.0f), n[1]},
        {vec4(bl[0], bl[1], bl[2], 1.0f), n[1]},
        {vec4(bl[0], bl[1], ur[2], 1.0f), n[1]},

        {vec4(ur[0], ur[1], ur[2], 1.0f), n[2]},
        {vec4(ur[0], ur[1], bl[2], 1.0f), n[2]},
        {vec4(bl[0], ur[1], bl[2], 1.0f), n[2]},
        {vec4(bl[0], ur[1], ur[2], 1.0f), n[2]},

        {vec4(ur[0], bl[1], ur[2], 1.0f), n[3]},
        {vec4(bl[0], bl[1], ur[2], 1.0f), n[3]},
        {vec4(bl[0], bl[1], bl[2], 1.0f), n[3]},
        {vec4(ur[0], bl[1], bl[2], 1.0f), n[3]},

        {vec4(ur[0], ur[1], ur[2], 1.0f), n[4]},
        {vec4(bl[0], ur[1], ur[2], 1.0f), n[4]},
        {vec4(bl[0], bl[1], ur[2], 1.0f), n[4]},
        {vec4(ur[0], bl[1], ur[2], 1.0f), n[4]},

        {vec4(ur[0], ur[1], bl[2], 1.0f), n[5]},
        {vec4(ur[0], bl[1], bl[2], 1.0f), n[5]},
        {vec4(bl[0], bl[1], bl[2], 1.0f), n[5]},
        {vec4(bl[0], ur[1], bl[2], 1.0f), n[5]},
    };

    VertexArray::Descriptor descs[] = {
        {GLSL_POSITION, VertexArray::TYPE_FLOAT,4, 0, false, 0},
        {GLSL_NORMAL,   VertexArray::TYPE_FLOAT,4,16, false, 0},
    };

    vbo.allocate(sizeof(v), VertexBufferObject::USAGE_STATIC, v);
    va.setAttributes(vbo, sizeof(vertex_t), descs, _countof(descs));
}


namespace {
    const float32 g_particle_par_volume = 30000.0; // particles / (1.0*1.0*1.0)
}

bool CreateCubeParticleSet( ParticleSet &pset, RigidInfo &ri, float32 half_len )
{
    SFMT random; random.initialize(3);

    float32 len = half_len*2.0f;
    vec4 pos = vec4(-half_len, -half_len, -half_len, 0.0f);
    float32 volume = len*len*len;
    uint32 num = static_cast<uint32>(volume * g_particle_par_volume);
    stl::vector<PSetParticle> particles;
    particles.resize(num);

    const vec4 planes[6] = {
        vec4( 1.0f, 0.0f, 0.0f, 0.0f),
        vec4(-1.0f, 0.0f, 0.0f, 0.0f),
        vec4( 0.0f, 1.0f, 0.0f, 0.0f),
        vec4( 0.0f,-1.0f, 0.0f, 0.0f),
        vec4( 0.0f, 0.0f, 1.0f, 0.0f),
        vec4( 0.0f, 0.0f,-1.0f, 0.0f),
    };
    for(uint32 i=0; i<num; ++i) {
        vec4 rv = vec4(random.genFloat32(),random.genFloat32(),random.genFloat32(),0.0f) * len;
        vec4 ppos = pos + rv;
        particles[i].position = ppos;
        particles[i].position.w = 1.0f;

        float32 max_d = 0.0f;
        uint32 max_p = 0;
        for(uint32 p=0; p<_countof(planes); ++p) {
            float32 d = glm::dot(ppos, planes[p]);
            if(d > max_d) {
                max_d = d;
                max_p = p;
            }
        }
        particles[i].normal = planes[max_p];
        particles[i].normal.w = max_d / half_len;
    }

    pset.setData(particles);
    ri.box_size = make_float4(half_len, half_len, half_len, 0.0f);
    return true;
}

bool CreateSphereParticleSet( ParticleSet &pset, RigidInfo &ri, float32 radius )
{
    SFMT random; random.initialize(5);

    vec4 half = vec4(0.5f, 0.5f, 0.5f, 0.0f);
    float32 volume = (4.0f/3.0f) * ist::PI * (radius*radius*radius);
    uint32 num = static_cast<uint32>(volume * g_particle_par_volume);
    stl::vector<PSetParticle> particles;
    particles.resize(num);

    for(uint32 i=0; i<num; ++i) {
        vec4 dir = glm::normalize(vec4(random.genFloat32(),random.genFloat32(),random.genFloat32(),0.0f)-half);
        float l = random.genFloat32()*radius;
        vec4 pos = dir*l;
        particles[i].position   = pos;
        particles[i].position.w = 1.0f;
        particles[i].normal     = dir;
        particles[i].normal.w   = l / radius;
    }

    pset.setData(particles);
    ri.sphere_radius = radius;
    return true;
}

} // namespace atomic
