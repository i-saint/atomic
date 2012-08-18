#ifndef __ist_UI_iuiCommon_h__
#define __ist_UI_iuiCommon_h__
#include "ist/Base.h"
#include "ist/Math.h"
#include "ist/GraphicsCommon/Image.h"

namespace ist {
namespace iui {

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



    enum UIEventType {
        UIE_MouseClick,
        UIE_MouseMove,
        UIE_KeyDown,
        UIE_KeyUp,
        UIE_IMEChar,
        UIE_IMEComplete,
    };

    class UIEvent : public SharedObject
    {
    public:
    };


} // namespace iui
} // namespace ist
#endif // __ist_UI_iuiCommon_h__
