#ifndef __atomic_Graphics_ResourceID__
#define __atomic_Graphics_ResourceID__
namespace atomic {

const size_t ATOMIC_MAX_CHARACTERS = 256;

enum SH_RID {
    SH_GBUFFER_FLOOR,
    SH_GBUFFER_FLUID,
    SH_GBUFFER_RIGID,
    SH_BLOODSTAIN,
    SH_DIRECTIONALLIGHT,
    SH_POINTLIGHT,
    SH_MICROSCOPIC,
    SH_FXAA_LUMA,
    SH_FXAA,
    SH_BLOOM_LUMINANCE,
    SH_BLOOM_HBLUR,
    SH_BLOOM_VBLUR,
    SH_BLOOM_COMPOSITE,
    SH_NORMAL_HBLUR,
    SH_NORMAL_VBLUR,
    SH_FADE,
    SH_FILL,
    SH_FILL_INSTANCED,
    SH_DISTANCE_FIELD,
    SH_OUTPUT,
    SH_DEBUG_SHOW_RGB,
    SH_DEBUG_SHOW_AAA,
    SH_END,
};

enum RT_RID {
    RT_GBUFFER,
    RT_GAUSS0,
    RT_GAUSS1,
    RT_OUTPUT0,
    RT_OUTPUT1,
    RT_GENERIC,
    RT_END,
};

enum SAMPLER_RID {
    SAMPLER_GBUFFER,
    SAMPLER_TEXTURE_DEFAULT,
    SAMPLER_END,
};

enum TEX1D_RID {
    TEX1D_DUMMY,
    TEX1D_END,
};

enum TEX2D_RID {
    TEX2D_RANDOM,
    TEX2D_ENTITY_PARAMS,
    TEX2D_END,
};

enum VA_RID {
    VA_SCREEN_QUAD,
    VA_FLOOR_QUAD,
    VA_BLOOM_LUMINANCE_QUADS,
    VA_BLOOM_BLUR_QUADS,
    VA_BLOOM_COMPOSITE_QUAD,
    VA_UNIT_CUBE,
    VA_UNIT_SPHERE,
    VA_FLUID_CUBE,
    VA_BLOOSTAIN_SPHERE,

    VA_FIELD_GRID,
    VA_DISTANCE_FIELD,

    VA_END,
};

enum VBO_RID {
    VBO_SCREEN_QUAD,
    VBO_FLOOR_QUAD,
    VBO_BLOOM_LUMINANCE_QUADS,
    VBO_BLOOM_BLUR_QUADS,
    VBO_BLOOM_COMPOSITE_QUAD,
    VBO_UNIT_CUBE,
    VBO_UNIT_SPHERE,
    VBO_FLUID_CUBE,
    VBO_BLOODSTAIN_SPHERE,
    VBO_BLOODSTAIN_PARTICLES,

    VBO_FLUID_PARTICLES,
    VBO_RIGID_PARTICLES,
    VBO_POINTLIGHT_INSTANCES,
    VBO_DIRECTIONALLIGHT_INSTANCES,

    VBO_FIELD_GRID,
    VBO_DISTANCE_FIELD_QUAD,
    VBO_DISTANCE_FIELD_POS,
    VBO_DISTANCE_FIELD_DIST,

    VBO_END,
};

enum IBO_RID {
    IBO_BLOODSTAIN_SPHERE,
    IBO_LIGHT_SPHERE,
    IBO_END,
};

enum UBO_RID {
    UBO_RENDER_STATES,
    UBO_FXAA_PARAMS,
    UBO_FADE_PARAMS,
    UBO_BLOOM_PARAMS,
    UBO_DEBUG_SHOW_BUFFER_PARAMS,
    UBO_END,
};

enum PSET_RID {
    PSET_CUBE_SMALL,
    PSET_CUBE_MEDIUM,
    PSET_CUBE_LARGE,
    PSET_SPHERE_SMALL,
    PSET_SPHERE_MEDIUM,
    PSET_SPHERE_LARGE,
    PSET_SPHERE_BULLET,
    PSET_INSTANCE,
    PSET_END,
};

} // namespace atomic
#endif //__atomic_Graphics_ResourceID__
