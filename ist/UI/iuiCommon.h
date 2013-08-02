#ifndef iui_Common_h
#define iui_Common_h
#include "ist/Base.h"
#include "ist/Math.h"
#include "ist/Window.h"

#if defined(iuiExportSymbols)
#   define iuiAPI istDLLExport
#elif defined(iuiImportSymbols)
#   define iuiAPI istDLLImport
#else
#   define iuiAPI
#endif // iuiExportSymbols


namespace iui {

using ist::int8;
using ist::int16;
using ist::int32;
using ist::int64;
using ist::uint8;
using ist::uint16;
using ist::uint32;
using ist::uint64;
using ist::uint128;
using ist::float16;
using ist::float32;
using ist::float64;
using ist::vec2;
using ist::vec3;
using ist::vec4;
using ist::mat2;
using ist::mat3;
using ist::mat4;
using ist::ivec2;
using ist::ivec3;
using ist::ivec4;
using ist::uvec2;
using ist::uvec3;
using ist::uvec4;
using ist::uvec2;
using ist::uvec3;
using ist::uvec4;
using ist::simdvec4;
using ist::simdmat4;
using ist::soavec24;
using ist::soavec34;
using ist::soavec44;
using ist::atomic_int32;

using ist::SharedObject;

typedef float32 Float;
typedef vec4 Color;
typedef vec2 Position;
typedef vec2 Size;
typedef stl::wstring String;


struct Range
{
    Float min, max;

    Range(Float mn=0.0f, Float mx=0.0f) : min(mn), max(mx) {}
};

struct Rect
{
    Position pos;
    Size size;

    Rect() {}
    Rect(const Position &p, const Size &s) : pos(p), size(s) {}
    const Position& getPosition() const { return pos; }
    const Size&     getSize() const     { return size; }
    void setPosition(const Position &v) { pos=v; }
    void setSize(const Size &v)         { size=v; }
};

struct Circle
{
    Position position;
    Float radius;
    Circle() : radius(0.0f) {}
    Circle(const Position& p, Float r) : position(p), radius(r) {}
    const Position& getPosition() const { return position; }
    Float getRadius() const             { return radius; }
    void setPosition(const Position &p) { position=p; }
    void setRadius(Float r)             { radius=r; }
};

struct Line
{
    Position begin, end;

    Position& operator[](size_t i) { reinterpret_cast<Position*>(this)[i]; }
    const Position& operator[](size_t i) const { reinterpret_cast<const Position*>(this)[i]; }
};

enum TextHAlign
{
    TA_HLeft,
    TA_HRight,
    TA_HCenter,
};
enum TextVAlign
{
    TA_VTop,
    TA_VBottom,
    TA_VCenter,
};


class UIRenderer;
class Widget;
class Style;
typedef ist::sv_set<Widget*, ist::less_id<Widget*> > WidgetCont;
typedef ist::vector<Style*> StyleCont;
typedef std::function<void (Widget*)> WidgetCallback;

} // namespace iui
#endif // iui_Common_h
