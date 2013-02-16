#ifndef ist_Base_Stringnize_h
#define ist_Base_Stringnize_h
#include "ist/Base/Types.h"
#include "ist/Base/Variant.h"

namespace ist {

stl::string Format(const char *str, ...);

void Stringnize(const bool      &v, stl::string &str);
void Stringnize(const int8      &v, stl::string &str);
void Stringnize(const int16     &v, stl::string &str);
void Stringnize(const int32     &v, stl::string &str);
void Stringnize(const uint8     &v, stl::string &str);
void Stringnize(const uint16    &v, stl::string &str);
void Stringnize(const uint32    &v, stl::string &str);
void Stringnize(const float32   &v, stl::string &str);
void Stringnize(const vec2      &v, stl::string &str);
void Stringnize(const vec3      &v, stl::string &str);
void Stringnize(const vec4      &v, stl::string &str);
void Stringnize(const ivec2     &v, stl::string &str);
void Stringnize(const ivec3     &v, stl::string &str);
void Stringnize(const ivec4     &v, stl::string &str);
void Stringnize(const variant16 &v, stl::string &str);

bool Parse(const char *str, bool    &v);
bool Parse(const char *str, int8    &v);
bool Parse(const char *str, int16   &v);
bool Parse(const char *str, int32   &v);
bool Parse(const char *str, uint8   &v);
bool Parse(const char *str, uint16  &v);
bool Parse(const char *str, uint32  &v);
bool Parse(const char *str, float32 &v);
bool Parse(const char *str, vec2    &v);
bool Parse(const char *str, vec3    &v);
bool Parse(const char *str, vec4    &v);
bool Parse(const char *str, ivec2   &v);
bool Parse(const char *str, ivec3   &v);
bool Parse(const char *str, ivec4   &v);
bool Parse(const char *str, variant16 &v);


} // namespace ist
#endif // ist_Base_Stringnize_h
