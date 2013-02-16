#include "istPCH.h"
#include "Stringnize.h"

namespace ist {



stl::string Format( const char *str, ... )
{
    char buf[256];
    va_list vl;
    va_start(vl, str);
    istVSNPrintf(buf, _countof(buf), str, vl);
    va_end(vl);
    return buf;
}


void Stringnize(const bool      &v, stl::string &str) { str = v ? "true" : "false"; }
void Stringnize(const int8      &v, stl::string &str) { str = Format("%i", (int32)v); }
void Stringnize(const int16     &v, stl::string &str) { str = Format("%i", (int32)v); }
void Stringnize(const int32     &v, stl::string &str) { str = Format("%i", (int32)v); }
void Stringnize(const uint8     &v, stl::string &str) { str = Format("%u", (uint32)v); }
void Stringnize(const uint16    &v, stl::string &str) { str = Format("%u", (uint32)v); }
void Stringnize(const uint32    &v, stl::string &str) { str = Format("%u", (uint32)v); }
void Stringnize(const float32   &v, stl::string &str) { str = Format("%f", v); }
void Stringnize(const vec2      &v, stl::string &str) { str = Format("%f,%f", v.x, v.y); }
void Stringnize(const vec3      &v, stl::string &str) { str = Format("%f,%f,%f", v.x, v.y, v.z); }
void Stringnize(const vec4      &v, stl::string &str) { str = Format("%f,%f,%f,%f", v.x, v.y, v.z, v.w); }
void Stringnize(const ivec2     &v, stl::string &str) { str = Format("%d,%d", v.x, v.y); }
void Stringnize(const ivec3     &v, stl::string &str) { str = Format("%d,%d,%d", v.x, v.y, v.z); }
void Stringnize(const ivec4     &v, stl::string &str) { str = Format("%d,%d,%d,%d", v.x, v.y, v.z, v.w); }
void Stringnize(const variant16 &v, stl::string &str)
{
    const uint8 *c = reinterpret_cast<const uint8*>(&v);
    str = Format("0x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x",
        c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[8],c[9],c[10],c[11],c[12],c[13],c[14],c[15] );
}



bool Parse( const char *str, bool &v )
{
    if(strncmp(str, "true", 4)==0 || strncmp(str, "bool(true)", 10)==0) { v=true; return true; }
    if(strncmp(str, "false", 5)==0 || strncmp(str, "bool(false)", 11)==0) { v=false; return true; }
    return false;
}

bool Parse( const char *str, int8 &v )
{
    int32 t;
    if(sscanf(str, "int8(%hhi)", &t)==1 || sscanf(str, "%hhi", &t)==1) { v=t; return true; }
    return false;
}

bool Parse( const char *str, int16 &v )
{
    int32 t;
    if(sscanf(str, "int16(%hi)", &t)==1 || sscanf(str, "%hi", &t)==1) { v=t; return true; }
    return false;
}

bool Parse( const char *str, int32 &v )
{
    int32 t;
    if(sscanf(str, "int32(%i)", &t)==1 || sscanf(str, "%i", &t)==1) { v=t; return true; }
    return false;
}

bool Parse( const char *str, uint8 &v )
{
    uint32 t;
    if(sscanf(str, "uint8(%hhu)", &t)==1 || sscanf(str, "%hhu", &t)==1) { v=t; return true; }
    return false;
}

bool Parse( const char *str, uint16 &v )
{
    uint32 t;
    if(sscanf(str, "uint16(%hu)", &t)==1 || sscanf(str, "%hu", &t)==1) { v=t; return true; }
    return false;
}

bool Parse( const char *str, uint32 &v )
{
    uint32 t;
    if(sscanf(str, "uint32(%u)", &t)==1 || sscanf(str, "%u", &t)==1) { v=t; return true; }
    return false;
}

bool Parse( const char *str, float32 &v )
{
    float32 t;
    if(sscanf(str, "float32(%f)", &t)==1 || sscanf(str, "%f", &t)==1) { v=t; return true; }
    return false;
}

bool Parse(const char *str, vec2 &v)
{
    vec2 t;
    if(sscanf(str, "vec2(%f,%f)", &t.x, &t.y)==2 || sscanf(str, "%f,%f", &t.x, &t.y)==2) { v=t; return true; }
    else if(sscanf(str, "vec2(%f)", &t.x)==1 || sscanf(str, "%f", &t.x)==1) { v=vec2(t.x); return true; }
    return false;
}

bool Parse(const char *str, vec3 &v)
{
    vec3 t;
    if(sscanf(str, "vec3(%f,%f,%f)", &t.x, &t.y, &t.z)==3 || sscanf(str, "%f,%f,%f", &t.x, &t.y, &t.z)==3) { v=t; return true; }
    else if(sscanf(str, "vec3(%f)", &t.x)==1 || sscanf(str, "%f", &t.x)==1) { v=vec3(t.x); return true; }
    return false;
}

bool Parse(const char *str, vec4 &v)
{
    vec4 t;
    if(sscanf(str, "vec4(%f,%f,%f,%f)", &t.x, &t.y, &t.z, &t.w)==4 || sscanf(str, "%f,%f,%f,%f", &t.x, &t.y, &t.z, &t.w)==4) { v=t; return true; }
    else if(sscanf(str, "vec4(%f,%f,%f)", &t.x, &t.y, &t.z)==3 || sscanf(str, "%f,%f,%f", &t.x, &t.y, &t.z)==3) { t.w=1.0f; v=t; return true; }
    else if(sscanf(str, "vec4(%f)", &t.x)==1 || sscanf(str, "%f", &t.x)==1) { v=vec4(t.x); return true; }
    return false;
}

bool Parse(const char *str, ivec2 &v)
{
    ivec2 t;
    if(sscanf(str, "ivec2(%d,%d)", &t.x, &t.y)==2 || sscanf(str, "%d,%d", &t.x, &t.y)==2) { v=t; return true; }
    else if(sscanf(str, "ivec2(%d)", &t.x)==1 || sscanf(str, "%d", &t.x)==1) { v=ivec2(t.x); return true; }
    return false;
}

bool Parse(const char *str, ivec3 &v)
{
    ivec3 t;
    if(sscanf(str, "ivec3(%d,%d,%d)", &t.x, &t.y, &t.z)==3 || sscanf(str, "%d,%d,%d", &t.x, &t.y, &t.z)==3) { v=t; return true; }
    else if(sscanf(str, "ivec3(%d)", &t.x)==1 || sscanf(str, "%d", &t.x)==1) { v=ivec3(t.x); return true; }
    return false;
}

bool Parse(const char *str, ivec4 &v)
{
    ivec4 t;
    if(sscanf(str, "ivec4(%d,%d,%d,%d)", &t.x, &t.y, &t.z, &t.w)==4 || sscanf(str, "%d,%d,%d,%d", &t.x, &t.y, &t.z, &t.w)==4) { v=t; return true; }
    else if(sscanf(str, "ivec4(%d)", &t.x)==1 || sscanf(str, "%d", &t.x)==1) { v=ivec4(t.x); return true; }
    return false;
}

bool Parse( const char *str, variant16 &v )
{
    const uint8 *c = reinterpret_cast<const uint8*>(&v);
    return
        Parse(str, v.cast<vec4>()) ||
        Parse(str, v.cast<vec3>()) ||
        Parse(str, v.cast<vec2>()) ||
        Parse(str, v.cast<bool>()) ||
        Parse(str, v.cast<float32>()) ||
        Parse(str, v.cast<int32>()) ||
        Parse(str, v.cast<uint32>()) ||
        sscanf(str, "0x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x",
            &c[0],&c[1],&c[2],&c[3],c[4],&c[5],&c[6],&c[7],&c[8],&c[9],&c[10],&c[11],&c[12],&c[13],&c[14],&c[15] )==16;
}

} // namespace ist
