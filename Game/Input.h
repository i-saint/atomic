#ifndef atomic_Game_Input_h
#define atomic_Game_Input_h
namespace atomic {

union LevelEditorCommand;


class InputState
{
public:
    struct istAlign(4) move_t {
        int16 x,y;

        move_t() : x(0),y(0) {}
        move_t(int16 _x, int16 _y) : x(_x) , y(_y) {}
        vec2 toF() const { return vec2(float32(x),float32(y))/32767.0f; }
    };
    typedef uint32 button_t;

    enum Direction {
        Dir_Left,
        Dir_Right,
        Dir_Up,
        Dir_Down,
    };

    InputState()
    {
        stl::fill_n(m_move, _countof(m_move), move_t());
        stl::fill_n(m_buttons, _countof(m_buttons), button_t());
    }

    const move_t& getRawMove() const { return m_move[0]; }
    vec2 getMove() const { return m_move[0].toF(); }
    bool isDirectionPressed(Direction d) const
    {
        switch(d) {
        case Dir_Right: return m_move[0].x>= 0.5f;
        case Dir_Left:  return m_move[0].x<=-0.5f;
        case Dir_Up:    return m_move[0].y>= 0.5f;
        case Dir_Down:  return m_move[0].y<=-0.5f;
        }
        return false;
    }
    bool isDirectionTriggered(Direction d) const
    {
        return isDirectionPressed(d) && (std::abs(m_move[1].x)<0.5f && std::abs(m_move[1].y)<0.5f);
    }

    int32 getButtons() const            { return m_buttons[0]; }
    bool isButtonPressed(int b) const   { return (m_buttons[0] & (1<<b)) !=0; }
    bool isButtonTriggered(int b) const { return isButtonPressed(b) && ((m_buttons[1] & (1<<b))==0); }

    void update(move_t p, button_t b)
    {
        m_move[1]   = m_move[0];
        m_buttons[1]= m_buttons[0];
        m_move[0]   = p;
        m_buttons[0]= b;
    }

private:
    move_t      m_move[2];
    button_t    m_buttons[2];
};

struct RepHeader
{
    struct {
        char magic[8];
        uint32 version;
        uint32 random_seed;
        uint32 total_frame;
        uint32 num_players;
        uint32 num_lecs;
    };

    RepHeader();
    bool isValid();
};

struct RepPlayer
{
    char name[32];
    uint32 equip;
    uint32 begin_frame;
    uint32 num_frame;

    RepPlayer();
};

struct RepInput
{
    InputState::move_t move;
    InputState::button_t buttons;
};






} // namespace atomic
#endif // atomic_Game_Input_h
