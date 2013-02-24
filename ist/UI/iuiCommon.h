#ifndef ist_UI_iuiCommon_h
#define ist_UI_iuiCommon_h
#include "ist/Base.h"
#include "ist/Math.h"
#include "ist/Graphics.h"
#include "ist/Window.h"

namespace ist {
namespace iui {

// とりあえず
namespace i3d = ::ist::i3dgl;


typedef float32 Float;
typedef RGBA_32F Color;
typedef vec2 Position;
typedef stl::wstring String;

typedef uint32 WidgetID;

struct Size
{
    Float width, height;

    Size(Float w=0.0f, Float h=0.0f) : width(w), height(h) {}
};

struct Range
{
    Float minimum, maximum;

    Range(Float mn=0.0f, Float mx=0.0f) : minimum(mn), maximum(mx) {}
};

struct Rect
{
    Position position;
    Size size;
};

struct Circle
{
    Position position;
    Float radius;
};

struct Line
{
    Position begin, end;
};



} // namespace iui
} // namespace ist
#endif // ist_UI_iuiCommon_h
