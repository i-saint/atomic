#include "atmPCH.h"
#include "LevelEditorCommand.h"

namespace atm {

void EntitiesQueryContext::clear()
{
    id.clear();
    trans.clear();
    size.clear();
    color.clear();
    bullets.clear();
    lasers.clear();
    fluids.clear();
}

size_t EntitiesQueryContext::sizeByte() const
{
    return
        sizeof(uint32)* id.size()       +
        sizeof(mat4)  * trans.size()    +
        sizeof(vec3)  * size.size()     +
        sizeof(vec4)  * color.size()    +
        sizeof(vec2)  * bullets.size()  +
        sizeof(vec3)  * lasers.size()   +
        sizeof(vec2)  * fluids.size();
}

void EntitiesQueryContext::makeArrayBuffer(stl::string &out)
{
    uint32 wpos = 0;

    uint32 num_entities = (uint32)id.size();
    uint32 num_bullets = (uint32)bullets.size();
    uint32 num_lasers = (uint32)lasers.size();
    uint32 num_fluids = (uint32)fluids.size();
    out.resize(sizeof(uint32)*4+sizeByte());

    *(uint32*)(&out[wpos]) = num_entities;
    wpos += sizeof(uint32);
    *(uint32*)(&out[wpos]) = num_bullets;
    wpos += sizeof(uint32);
    *(uint32*)(&out[wpos]) = num_lasers;
    wpos += sizeof(uint32);
    *(uint32*)(&out[wpos]) = num_fluids;
    wpos += sizeof(uint32);

    if(num_entities) {

        memcpy(&out[wpos], &id[0], sizeof(id)*num_entities);
        wpos += sizeof(uint32)*num_entities;
        memcpy(&out[wpos], &trans[0], sizeof(mat4)*num_entities);
        wpos += sizeof(mat4)*num_entities;
        memcpy(&out[wpos], &size[0], sizeof(vec3)*num_entities);
        wpos += sizeof(vec3)*num_entities;
        memcpy(&out[wpos], &color[0], sizeof(vec4)*num_entities);
        wpos += sizeof(vec4)*num_entities;
    }
    if(num_bullets) {
        memcpy(&out[wpos], &bullets[0], sizeof(vec2)*num_bullets);
        wpos += sizeof(vec2)*num_bullets;
    }
    if(num_lasers) {
        memcpy(&out[wpos], &lasers[0], sizeof(vec3)*num_lasers);
        wpos += sizeof(vec3)*num_lasers;
    }
    if(num_fluids) {
        memcpy(&out[wpos], &fluids[0], sizeof(vec2)*num_fluids);
        wpos += sizeof(vec2)*num_lasers;
    }
}

} // namespace atm
