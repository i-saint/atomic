#ifndef __ist_UI_iuiRenderer_h__
#define __ist_UI_iuiRenderer_h__
#include "iuiCommon.h"
#include "ist/GraphicsGL.h"

namespace ist {
namespace iui {

    enum RenderCommandType {
        RC_Rect,
        RC_Circle,
        RC_Line,
        RC_Text,
    };
    enum FillMode {
        FM_Fill,
        FM_Line,
    };

    class IRenderCommand;
    typedef stl::vector<IRenderCommand> RenderCommands;


    class istInterModule IRenderCommand
    {
    public:
        virtual ~IRenderCommand() {}
        virtual RenderCommandType getType() const=0;
    };

    class istInterModule RCRect : public IRenderCommand
    {
    public:
        RCRectFill() {}
        virtual RenderCommandType getType() const { return RC_Rect; }

        Color color;
        FillMode fillmode;
        Rect rect;
    };

    class istInterModule RCCircle : public IRenderCommand
    {
    public:
        RCRectFill() {}
        virtual RenderCommandType getType() const { return RC_Circle; }

        Color color;
        FillMode fillmode;
        Circle rect;
    };

    class istInterModule RCLine : public IRenderCommand
    {
    public:
        RCRectFill() {}
        virtual RenderCommandType getType() const { return RC_Line; }

        Color color;
        Line rect;
    };

    class istInterModule RCText : public IRenderCommand
    {
    public:
        RCRectFill() {}
        virtual RenderCommandType getType() const { return RC_Line; }

        Color color;
        Position position;
        String text;
    };



    class istInterModule UIRenderer : public SharedObject
    {
    public:
        UIRenderer();
        ~UIRenderer();

        void addCommand(IRenderCommand *command);
        void flush();

    private:
        RenderCommands m_commands;
    };

} // namespace iui
} // namespace ist
#endif // __ist_UI_iuiRenderer_h__
