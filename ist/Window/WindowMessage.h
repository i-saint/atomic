#ifndef __ist_Application_WindowMessage__
#define __ist_Application_WindowMessage__
namespace ist {

struct WindowMessage
{
    enum TYPE
    {
        MES_CLOSE,
        MES_ACTIVE,
        MES_KEYBOARD,
        MES_MOUSE,
        MES_JOYSTICK,
        MES_WINDOW_SIZE,
        MES_WINDOW_MOVE,
        MES_FOCUS,
        MES_IME_BEGIN,
        MES_IME_END,
        MES_IME_CANDIDATE_OPEN,
        MES_IME_CANDIDATE_CLOSE,
        MES_IME_CANDIDATE_CHANGE,
        MES_IME_CURSOR_MOVE,
        MES_IME_CHAR,
        MES_IME_RESULT,
    };

    int type;
};

struct WM_Close : public WindowMessage
{
};

struct WM_Active : public WindowMessage
{
    enum STATE
    {
        ST_ACTIVATED,
        ST_DEACTIVATED,
    };
    short state;
};

struct WM_WindowSize : public WindowMessage
{
    ivec2 window_size;
};

struct WM_WindowMove : public WindowMessage
{
    ivec2 window_pos;
};


struct WM_Keyboard : public WindowMessage
{
    enum ACTION
    {
        ACT_KEYUP,
        ACT_KEYDOWN,
        ACT_CHAR,
    };
    enum KEY
    {
        KEY_ESCAPE = VK_ESCAPE,
    };

    short action;
    short key;
};

struct WM_Mouse : public WindowMessage
{
    enum ACTION
    {
        ACT_BUTTON_UP,
        ACT_BUTTON_DOWN,
        ACT_MOVE,
    };
    enum BUTTON
    {
        BU_LEFT     = 0x01,
        BU_RIGHT    = 0x02,
        BU_MIDDLE   = 0x10,
    };
    enum CONTROL
    {
        CT_CONTROL  = 0x08,
        CT_SHIFT    = 0x04,
    };

    short action;
    short button;
    short control;
    short x;
    short y;
};

struct WM_IME : public WindowMessage
{
    size_t text_len;
    size_t num_candidates;
    size_t cursor_pos;
    size_t candidate_index;
    wchar_t *text;
    wchar_t *candidates;

    void initialize() {
        text_len = 0;
        num_candidates = 0;
        cursor_pos = 0;
        candidate_index = 0;
        text = NULL;
        candidates = NULL;
    }
};


} // namspace ist
#endif // __ist_Application_WindowMessage__
