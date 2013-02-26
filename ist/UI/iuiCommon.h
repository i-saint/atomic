#ifndef iui_Common_h
#define iui_Common_h
#include "ist/Base.h"
#include "ist/Math.h"
#include "ist/Window.h"

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
using ist::deep_copy_ptr;
using ist::WM_Base;
using ist::WM_Mouse;
using ist::WM_Keyboard;
using ist::WM_IME;

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


typedef uint32 WidgetID;

class UIRenderer;

class Widget;
class Style;
typedef ist::vector<Widget*> WidgetCont;
typedef std::function<void (Widget*)> WidgetCallback;

class Panel;
class Window;

class Button;
class ToggleButton;
class Checkbox;

class Editbox;
class EditboxMultiline;


} // namespace iui
#endif // iui_Common_h
