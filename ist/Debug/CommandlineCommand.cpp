#include "istPCH.h"
#include "CommandlineCommand.h"

namespace ist {

bool CLParseArg( const char *str, bool &v )
{
    if(strncmp(str, "true", 4)==0 || strncmp(str, "bool(true)", 10)==0) { v=true; return true; }
    if(strncmp(str, "false", 5)==0 || strncmp(str, "bool(false)", 11)==0) { v=false; return true; }
    return false;
}

bool CLParseArg( const char *str, int8 &v )
{
    int32 t;
    if(sscanf(str, "int8(%hhi)", &t)==1 || sscanf(str, "%hhi", &t)==1) { v=t; return true; }
    return false;
}

bool CLParseArg( const char *str, int16 &v )
{
    int32 t;
    if(sscanf(str, "int16(%hi)", &t)==1 || sscanf(str, "%hi", &t)==1) { v=t; return true; }
    return false;
}

bool CLParseArg( const char *str, int32 &v )
{
    int32 t;
    if(sscanf(str, "int32(%i)", &t)==1 || sscanf(str, "%i", &t)==1) { v=t; return true; }
    return false;
}

bool CLParseArg( const char *str, uint8 &v )
{
    uint32 t;
    if(sscanf(str, "uint8(%hhu)", &t)==1 || sscanf(str, "%hhu", &t)==1) { v=t; return true; }
    return false;
}

bool CLParseArg( const char *str, uint16 &v )
{
    uint32 t;
    if(sscanf(str, "uint16(%hu)", &t)==1 || sscanf(str, "%hu", &t)==1) { v=t; return true; }
    return false;
}

bool CLParseArg( const char *str, uint32 &v )
{
    uint32 t;
    if(sscanf(str, "uint32(%u)", &t)==1 || sscanf(str, "%u", &t)==1) { v=t; return true; }
    return false;
}

bool CLParseArg( const char *str, float32 &v )
{
    float32 t;
    if(sscanf(str, "float32(%f)", &t)==1 || sscanf(str, "%f", &t)==1) { v=t; return true; }
    return false;
}

bool CLParseArg(const char *str, vec2 &v)
{
    vec2 t;
    if(sscanf(str, "vec2(%f,%f)", &t.x, &t.y)==2 || sscanf(str, "%f,%f", &t.x, &t.y)==2) { v=t; return true; }
    else if(sscanf(str, "vec2(%f)", &t.x)==1 || sscanf(str, "%f", &t.x)==1) { v=vec2(t.x); return true; }
    return false;
}

bool CLParseArg(const char *str, vec3 &v)
{
    vec3 t;
    if(sscanf(str, "vec3(%f,%f,%f)", &t.x, &t.y, &t.z)==3 || sscanf(str, "%f,%f,%f", &t.x, &t.y, &t.z)==3) { v=t; return true; }
    else if(sscanf(str, "vec3(%f)", &t.x)==1 || sscanf(str, "%f", &t.x)==1) { v=vec3(t.x); return true; }
    return false;
}

bool CLParseArg(const char *str, vec4 &v)
{
    vec4 t;
    if(sscanf(str, "vec4(%f,%f,%f,%f)", &t.x, &t.y, &t.z, &t.w)==4 || sscanf(str, "%f,%f,%f,%f", &t.x, &t.y, &t.z, &t.w)==4) { v=t; return true; }
    else if(sscanf(str, "vec4(%f,%f,%f)", &t.x, &t.y, &t.z)==3 || sscanf(str, "%f,%f,%f", &t.x, &t.y, &t.z)==3) { t.w=1.0f; v=t; return true; }
    else if(sscanf(str, "vec4(%f)", &t.x)==1 || sscanf(str, "%f", &t.x)==1) { v=vec4(t.x); return true; }
    return false;
}

bool CLParseArg(const char *str, ivec2 &v)
{
    ivec2 t;
    if(sscanf(str, "ivec2(%d,%d)", &t.x, &t.y)==2 || sscanf(str, "%d,%d", &t.x, &t.y)==2) { v=t; return true; }
    else if(sscanf(str, "ivec2(%d)", &t.x)==1 || sscanf(str, "%d", &t.x)==1) { v=ivec2(t.x); return true; }
    return false;
}

bool CLParseArg(const char *str, ivec3 &v)
{
    ivec3 t;
    if(sscanf(str, "ivec3(%d,%d,%d)", &t.x, &t.y, &t.z)==3 || sscanf(str, "%d,%d,%d", &t.x, &t.y, &t.z)==3) { v=t; return true; }
    else if(sscanf(str, "ivec3(%d)", &t.x)==1 || sscanf(str, "%d", &t.x)==1) { v=ivec3(t.x); return true; }
    return false;
}

bool CLParseArg(const char *str, ivec4 &v)
{
    ivec4 t;
    if(sscanf(str, "ivec4(%d,%d,%d,%d)", &t.x, &t.y, &t.z, &t.w)==4 || sscanf(str, "%d,%d,%d,%d", &t.x, &t.y, &t.z, &t.w)==4) { v=t; return true; }
    else if(sscanf(str, "ivec4(%d)", &t.x)==1 || sscanf(str, "%d", &t.x)==1) { v=ivec4(t.x); return true; }
    return false;
}

bool CLParseArg( const char *str, Variant16 &v )
{
    return
        CLParseArg(str, v.cast<vec4>()) ||
        CLParseArg(str, v.cast<vec3>()) ||
        CLParseArg(str, v.cast<vec2>()) ||
        CLParseArg(str, v.cast<bool>()) ||
        CLParseArg(str, v.cast<float32>()) ||
        CLParseArg(str, v.cast<int32>()) ||
        CLParseArg(str, v.cast<uint32>());
}

} // namespace ist
