#include "stdafx.h"
#include "ist/ist.h"
#include "types.h"
#include "Util.h"
#include "Graphics/AtomicRenderingSystem.h"
#include "Graphics/ResourceID.h"
#include "Graphics/ParticleSet.h"
#include "Graphics/CreateModelData.h"
#include "shader/Semantics.h"
#include "psym/psym.h"
#include <math.h>


namespace atomic {


void CreateFloorQuad( VertexArray *va, Buffer *&vbo, vec4 pos, vec4 size )
{
    struct istAlign(16) vertex_t
    {
        vec4 pos;
        vec4 normal;
        vec2 texcoord;
    };

    vertex_t vertices[4] = {
        {vec4( size.x, size.y, size.z, 1.0f)+pos, vec4(0.0f, 0.0f, 1.0f, 0.0f), vec2(1.0f, 1.0f)},
        {vec4(   0.0f, size.y, size.z, 1.0f)+pos, vec4(0.0f, 0.0f, 1.0f, 0.0f), vec2(0.0f, 1.0f)},
        {vec4(   0.0f,   0.0f, size.z, 1.0f)+pos, vec4(0.0f, 0.0f, 1.0f, 0.0f), vec2(0.0f, 0.0f)},
        {vec4( size.x,   0.0f, size.z, 1.0f)+pos, vec4(0.0f, 0.0f, 1.0f, 0.0f), vec2(1.0f, 0.0f)},
    };

    VertexDesc descs[] = {
        {GLSL_POSITION, I3D_FLOAT,4,  0, false, 0},
        {GLSL_NORMAL,   I3D_FLOAT,4, 16, false, 0},
        {GLSL_TEXCOORD1,I3D_FLOAT,2, 32, false, 0},
    };

    i3d::Device *dev = atomicGetGLDevice();
    vbo = CreateVertexBuffer(dev, sizeof(vertex_t)*_countof(vertices), I3D_USAGE_STATIC, vertices);
    va->setAttributes(*vbo, sizeof(vertex_t), descs, _countof(descs));
}


void CreateSphere(
    VertexArray *va, Buffer *&vbo, Buffer *&ibo,
    float32 radius, uint32 div_xz, uint32 div_y)
{
    struct istAlign(16) vertex_t
    {
        vec4 pos;
        vec4 normal;
    };
    const uint32 vertex_size = sizeof(vertex_t)*div_y*div_xz;
    const uint32 index_size = sizeof(uint32)*(div_y-1)*(div_xz)*4;
    vertex_t *vert = (vertex_t*)istAlignedMalloc(vertex_size, 16);
    uint32 *index = (uint32*)istAlignedMalloc(index_size, 16);

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

    VertexDesc descs[] = {
        {GLSL_POSITION, I3D_FLOAT,4,  0, false, 0},
        {GLSL_NORMAL,   I3D_FLOAT,4, 16, false, 0},
    };

    i3d::Device *dev = atomicGetGLDevice();
    vbo = CreateVertexBuffer(dev, vertex_size, I3D_USAGE_STATIC, vert);
    ibo = CreateIndexBuffer(dev, index_size, I3D_USAGE_STATIC, index);
    va->setAttributes(*vbo, sizeof(vertex_t), descs, _countof(descs));

    istFree(index);
    istFree(vert);
}

void CreateScreenQuad( VertexArray *va, Buffer *&vbo )
{
    struct istAlign(16) vertex_t
    {
        vec2 pos;
        vec2 tex;
    } v[4] = {
        {vec2( 1.0f, 1.0f), vec2(1.0, 1.0)},
        {vec2(-1.0f, 1.0f), vec2(0.0, 1.0)},
        {vec2(-1.0f,-1.0f), vec2(0.0, 0.0)},
        {vec2( 1.0f,-1.0f), vec2(1.0, 0.0)},
    };
    VertexDesc descs[] = {
        {GLSL_POSITION,  I3D_FLOAT,2, 0, false, 0},
        {GLSL_TEXCOORD0, I3D_FLOAT,2, 8, false, 0},
    };

    i3d::Device *dev = atomicGetGLDevice();
    vbo = CreateVertexBuffer(dev, sizeof(v), I3D_USAGE_STATIC, v);
    va->setAttributes(*vbo, sizeof(vertex_t), descs, _countof(descs));
}

void CreateBloomLuminanceQuads( VertexArray *va, Buffer *&vbo )
{
    struct istAlign(16) vertex_t
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
    VertexDesc descs[] = {
        {GLSL_POSITION,  I3D_FLOAT,2, 0, false, 0},
        {GLSL_TEXCOORD0, I3D_FLOAT,2, 8, false, 0},
    };

    i3d::Device *dev = atomicGetGLDevice();
    vbo = CreateVertexBuffer(dev, sizeof(v), I3D_USAGE_STATIC, v);
    va->setAttributes(*vbo, sizeof(vertex_t), descs, _countof(descs));
}

void CreateBloomBlurQuads( VertexArray *va, Buffer *&vbo )
{
    const vec2 tp[4] = {vec2(0.0, 0.0), vec2(0.5,  0.0), vec2(0.75,  0.0 ), vec2(0.875,  0.0)};
    const vec2 ts[4] = {vec2(0.5, 1.0), vec2(0.25, 0.5), vec2(0.125, 0.25), vec2(0.0625, 0.125)};

    struct istAlign(16) vertex_t
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
    VertexDesc descs[] = {
        {GLSL_POSITION,  I3D_FLOAT,2, 0, false, 0},
        {GLSL_TEXCOORD0, I3D_FLOAT,2, 8, false, 0},
        {GLSL_TEXCOORD1, I3D_FLOAT,2,16, false, 0},
        {GLSL_TEXCOORD2, I3D_FLOAT,2,24, false, 0},
    };

    i3d::Device *dev = atomicGetGLDevice();
    vbo = CreateVertexBuffer(dev, sizeof(v), I3D_USAGE_STATIC, v);
    va->setAttributes(*vbo, sizeof(vertex_t), descs, _countof(descs));
}

void CreateBloomCompositeQuad( VertexArray *va, Buffer *&vbo )
{
    const vec2 tp[4] = {vec2(0.0, 0.0), vec2(0.5,  0.0), vec2(0.75,  0.0 ), vec2(0.875,  0.0)};
    const vec2 ts[4] = {vec2(0.5, 1.0), vec2(0.25, 0.5), vec2(0.125, 0.25), vec2(0.0625, 0.125)};

    struct istAlign(16) vertex_t
    {
        vec2 pos;
        vec2 tex[4];
    } v[4] = {
        {vec2( 1.0, 1.0), {tp[0]+ts[0], tp[1]+ts[1], tp[2]+ts[2], tp[3]+ts[3]}},
        {vec2(-1.0, 1.0), {tp[0]+vec2(0.0,ts[0].y), tp[1]+vec2(0.0,ts[1].y), tp[2]+vec2(0.0,ts[2].y), tp[3]+vec2(0.0,ts[3].y)}},
        {vec2(-1.0,-1.0), {tp[0], tp[1], tp[2], tp[3]}},
        {vec2( 1.0,-1.0), {tp[0]+vec2(ts[0].x,0.0), tp[1]+vec2(ts[1].x,0.0), tp[2]+vec2(ts[2].x,0.0), tp[3]+vec2(ts[3].x,0.0)}},
    };

    VertexDesc descs[] = {
        {GLSL_POSITION,  I3D_FLOAT,2, 0, false, 0},
        {GLSL_TEXCOORD0, I3D_FLOAT,2, 8, false, 0},
        {GLSL_TEXCOORD1, I3D_FLOAT,2,16, false, 0},
        {GLSL_TEXCOORD2, I3D_FLOAT,2,24, false, 0},
        {GLSL_TEXCOORD3, I3D_FLOAT,2,32, false, 0},
    };

    i3d::Device *dev = atomicGetGLDevice();
    vbo = CreateVertexBuffer(dev, sizeof(v), I3D_USAGE_STATIC, v);
    va->setAttributes(*vbo, sizeof(vertex_t), descs, _countof(descs));
}

void CreateCube( VertexArray *va, Buffer *&vbo, float32 half_len )
{
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

    struct istAlign(16) vertex_t
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

    VertexDesc descs[] = {
        {GLSL_POSITION, I3D_FLOAT,4, 0, false, 0},
        {GLSL_NORMAL,   I3D_FLOAT,4,16, false, 0},
    };

    i3d::Device *dev = atomicGetGLDevice();
    vbo = CreateVertexBuffer(dev, sizeof(v), I3D_USAGE_STATIC, v);
    va->setAttributes(*vbo, sizeof(vertex_t), descs, _countof(descs));
}

void CreateFieldGridLines( VertexArray *va, Buffer *&vbo )
{
    struct vertex_t
    {
        vec4 pos;
        vec4 color;
    };
    stl::vector<vertex_t> vertices;

    vec3 div    = vec3(PSYM_GRID_DIV, PSYM_GRID_DIV, PSYM_GRID_DIV);
    vec3 bl     = vec3(-PSYM_GRID_SIZE*0.5f, -PSYM_GRID_SIZE*0.5f, 0.0f);
    vec3 size   = vec3(PSYM_GRID_SIZE);
    vec3 cell   = size / div;
    vertices.reserve((PSYM_GRID_DIV+1) * (PSYM_GRID_DIV+1) * 2);

    vec4 color1 = vec4(0.25f, 0.5f, 1.0f, 0.1f);
    vec4 color2 = vec4(0.25f, 0.5f, 1.0f, 0.15f);
    for(uint32 xi=0; xi<=div.x; ++xi) {
        vertex_t t[2];
        t[0].pos = vec4(bl + vec3(cell.x*xi,   0.0f, 0.0f), 1.0f);
        t[1].pos = vec4(bl + vec3(cell.x*xi, size.y, 0.0f), 1.0f);
        vec4 color = xi %4 == 0 ? color2 : color1;
        t[0].color = color;
        t[1].color = color;
        vertices.insert(vertices.end(), t, t+_countof(t));
    }
    for(uint32 yi=0; yi<=div.y; ++yi) {
        vertex_t t[2];
        t[0].pos = vec4(bl + vec3(  0.0f, cell.y*yi, 0.0f), 1.0f);
        t[1].pos = vec4(bl + vec3(size.x, cell.y*yi, 0.0f), 1.0f);
        vec4 color = yi %4 == 0 ? color2 : color1;
        t[0].color = color;
        t[1].color = color;
        vertices.insert(vertices.end(), t, t+_countof(t));
   }

    for(uint32 i=0; i<vertices.size(); ++i) {
    }

    VertexDesc descs[] = {
        {GLSL_POSITION, I3D_FLOAT,4, 0, false, 0},
        {GLSL_COLOR,    I3D_FLOAT,4,16, false, 0},
    };

    i3d::Device *dev = atomicGetGLDevice();
    vbo = CreateVertexBuffer(dev, sizeof(vertex_t)*vertices.size(), I3D_USAGE_STATIC, &vertices[0]);
    va->setAttributes(*vbo, sizeof(vertex_t), descs, _countof(descs));
}

void CreateDistanceFieldQuads( VertexArray *va, Buffer *&quad_model, Buffer *&quad_pos, Buffer *&quad_dist )
{
    vec3 div    = vec3(PSYM_GRID_DIV, PSYM_GRID_DIV, PSYM_GRID_DIV);
    vec3 bl     = vec3(-PSYM_GRID_SIZE*0.5f, -PSYM_GRID_SIZE*0.5f, 0.0f);
    vec3 size   = vec3(PSYM_GRID_SIZE);
    vec3 cell   = size / div;

    {
        struct istAlign(16) vertex_t
        {
            vec4 pos;
        } vertices[4] = {
            vec4(  0.0f,   0.0f, 0.0f, 1.0f),
            vec4(cell.x,   0.0f, 0.0f, 1.0f),
            vec4(cell.x, cell.y, 0.0f, 1.0f),
            vec4(  0.0f, cell.y, 0.0f, 1.0f),
        };
        VertexDesc descs[] = {
            {GLSL_POSITION, I3D_FLOAT,4, 0, false, 0},
        };

        i3d::Device *dev = atomicGetGLDevice();
        quad_model = CreateVertexBuffer(dev, sizeof(vertex_t)*_countof(vertices), I3D_USAGE_STATIC, vertices);
        if(quad_model && va) { va->setAttributes(*quad_model, sizeof(vertex_t), descs, _countof(descs)); }
    }

    {
        struct vertex_t
        {
            vec4 pos;
        };
        stl::vector<vertex_t> vertices;
        vertices.reserve(PSYM_GRID_DIV*PSYM_GRID_DIV);
        for(uint32 yi=0; yi<div.y; ++yi) {
            for(uint32 xi=0; xi<div.x; ++xi) {
                vertex_t t = { vec4(bl+vec3(cell.x*xi, cell.y*yi, 0.0f), 0.0f) };
                vertices.push_back(t);
            }
        }
        VertexDesc descs[] = {
            {GLSL_INSTANCE_POSITION, I3D_FLOAT,4, 0, false, 1},
        };

        i3d::Device *dev = atomicGetGLDevice();
        quad_pos = CreateVertexBuffer(dev, sizeof(vertex_t)*vertices.size(), I3D_USAGE_STATIC, &vertices[0]);
        va->setAttributes(*quad_pos, sizeof(vertex_t), descs, _countof(descs));
    }
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
        particles[i].position = vec3(ppos);

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
    assign_float4(ri.box_size, half_len, half_len, half_len, 0.0f);
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
        particles[i].position   = vec3(pos);
        particles[i].normal     = dir;
        particles[i].normal.w   = l / radius;
    }

    pset.setData(particles);
    ri.sphere_radius = radius;
    return true;
}

bool CreateBulletParticleSet( ParticleSet &pset, RigidInfo &ri )
{
    float32 r = 0.03f * 0.5f;
    vec4 center = vec4();
    mat4 z120 = glm::rotate(mat4(), 120.0f, vec3(0.0f, 0.0f, 1.0f));
    const vec4 pos[5] = {
        vec4(0.0f, 0, 0.0f, 1.0f),
        vec4(0.0f, r, 0.0f, 1.0f),
        glm::rotate(mat4(),  30.0f, vec3(0.0f, 1.0f, 0.0f)) * (z120*vec4(0.0f, r, 0.0f, 1.0f)),
        glm::rotate(mat4(), 150.0f, vec3(0.0f, 1.0f, 0.0f)) * (z120*vec4(0.0f, r, 0.0f, 1.0f)),
        glm::rotate(mat4(), 270.0f, vec3(0.0f, 1.0f, 0.0f)) * (z120*vec4(0.0f, r, 0.0f, 1.0f)),
    };
    const vec4 normal[5] = {
        vec4(0.0f, 0.0f, 0.0f, 0.6f),
        vec4(glm::normalize(vec3(pos[1])), 0.9f),
        vec4(glm::normalize(vec3(pos[2])), 0.9f),
        vec4(glm::normalize(vec3(pos[3])), 0.9f),
        vec4(glm::normalize(vec3(pos[4])), 0.9f),
    };

    stl::vector<PSetParticle> particles;
    particles.resize(_countof(pos));
    for(uint32 i=0; i<_countof(pos); ++i) {
        particles[i].normal = normal[i];
        particles[i].position = vec3(pos[i]);
    }

    pset.setData(particles);
    ri.sphere_radius = 0.03f;
    return true;
}

} // namespace atomic
