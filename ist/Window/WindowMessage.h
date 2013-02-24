#ifndef ist_Application_WindowMessage_h
#define ist_Application_WindowMessage_h
namespace ist {

struct istInterModule WindowMessage
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

struct istInterModule WM_Close : public WindowMessage
{
};

struct istInterModule WM_Active : public WindowMessage
{
    enum STATE
    {
        ST_ACTIVATED,
        ST_DEACTIVATED,
    };
    short state;
};

struct istInterModule WM_WindowSize : public WindowMessage
{
    ivec2 window_size;
};

struct istInterModule WM_WindowMove : public WindowMessage
{
    ivec2 window_pos;
};


struct istInterModule WM_Keyboard : public WindowMessage
{
    enum ACTION
    {
        ACT_KEYUP,
        ACT_KEYDOWN,
        ACT_CHAR,
    };

    short action;
    short key;
};

struct istInterModule WM_Mouse : public WindowMessage
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

    uint16 x;
    uint16 y;
    uint8 action;
    uint8 button;
    uint8 control;
};



// MES_IME_CHAR, MES_IME_RESULT のとき、text, text_len に入力データが入っている
struct istInterModule WM_IME : public WindowMessage
{
    uint32 text_len;
    uint32 num_candidates;
    uint32 cursor_pos;
    uint32 candidate_index;
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
#endif // ist_Application_WindowMessage_h
