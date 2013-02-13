#include "istPCH.h"
#include "CommandlineCommand.h"

namespace ist {

bool CLParseArg( const char *str, int8 &v )
{
    int32 t;
    if(sscanf(str, "%hhi", &t)==1) { v=t; return true; }
    return false;
}

bool CLParseArg( const char *str, int16 &v )
{
    int32 t;
    if(sscanf(str, "%hi", &t)==1) { v=t; return true; }
    return false;
}

bool CLParseArg( const char *str, int32 &v )
{
    int32 t;
    if(sscanf(str, "%i", &t)==1) { v=t; return true; }
    return false;
}

bool CLParseArg( const char *str, uint8 &v )
{
    uint32 t;
    if(sscanf(str, "%hhu", &t)==1) { v=t; return true; }
    return false;
}

bool CLParseArg( const char *str, uint16 &v )
{
    uint32 t;
    if(sscanf(str, "%hu", &t)==1) { v=t; return true; }
    return false;
}

bool CLParseArg( const char *str, uint32 &v )
{
    uint32 t;
    if(sscanf(str, "%u", &t)==1) { v=t; return true; }
    return false;
}

bool CLParseArg( const char *str, float32 &v )
{
    float32 t;
    if(sscanf(str, "%f", &t)==1) { v=t; return true; }
    return false;
}

bool CLParseArg(const char *str, vec2 &v)
{
    vec2 t;
    if(sscanf(str, "vec2(%f,%f)", &t.x, &t.y)==2) { v=t; return true; }
    else if(sscanf(str, "vec2(%f)", &t.x)==1) { v=vec2(t.x); return true; }
    return false;
}

bool CLParseArg(const char *str, vec3 &v)
{
    vec3 t;
    if(sscanf(str, "vec3(%f,%f,%f)", &t.x, &t.y, &t.z)==3) { v=t; return true; }
    else if(sscanf(str, "vec3(%f)", &t.x)==1) { v=vec3(t.x); return true; }
    return false;
}

bool CLParseArg(const char *str, vec4 &v)
{
    vec4 t;
    if(sscanf(str, "vec4(%f,%f,%f,%f)", &t.x, &t.y, &t.z, &t.w)==4) { v=t; return true; }
    else if(sscanf(str, "vec4(%f)", &t.x)==1) { v=vec4(t.x); return true; }
    return false;
}

bool CLParseArg(const char *str, ivec2 &v)
{
    ivec2 t;
    if(sscanf(str, "ivec2(%d,%d)", &t.x, &t.y)==2) { v=t; return true; }
    else if(sscanf(str, "ivec2(%d)", &t.x)==1) { v=ivec2(t.x); return true; }
    return false;
}

bool CLParseArg(const char *str, ivec3 &v)
{
    ivec3 t;
    if(sscanf(str, "ivec3(%d,%d,%d)", &t.x, &t.y, &t.z)==3) { v=t; return true; }
    else if(sscanf(str, "ivec3(%d)", &t.x)==1) { v=ivec3(t.x); return true; }
    return false;
}

bool CLParseArg(const char *str, ivec4 &v)
{
    ivec4 t;
    if(sscanf(str, "ivec4(%d,%d,%d,%d)", &t.x, &t.y, &t.z, &t.w)==4) { v=t; return true; }
    else if(sscanf(str, "ivec4(%d)", &t.x)==1) { v=ivec4(t.x); return true; }
    return false;
}

} // namespace ist
