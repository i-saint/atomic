#ifndef atomic_Game_Input_h
#define atomic_Game_Input_h
namespace atomic {

union LevelEditorCommand;


class InputState
{
public:
    enum DIRECTION {
        DIR_LEFT,
        DIR_RIGHT,
        DIR_UP,
        DIR_DOWN,
    };

    InputState()
    {
        stl::fill_n(m_move, _countof(m_move), vec2(0.0f));
        stl::fill_n(m_buttons, _countof(m_buttons), 0);
    }

    vec2 getMove() const                { return m_move[0]; }
    bool isDirectionPressed(DIRECTION d) const
    {
        switch(d) {
        case DIR_RIGHT: return m_move[0].x>= 0.5f;
        case DIR_LEFT:  return m_move[0].x<=-0.5f;
        case DIR_UP:    return m_move[0].y>= 0.5f;
        case DIR_DOWN:  return m_move[0].y<=-0.5f;
        }
        return false;
    }
    bool isDirectionTriggered(DIRECTION d) const
    {
        return isDirectionPressed(d) && (std::abs(m_move[1].x)<0.5f && std::abs(m_move[1].y)<0.5f);
    }

    int32 getButtons() const            { return m_buttons[0]; }
    bool isButtonPressed(int b) const   { return (m_buttons[0] & (1<<b)) !=0; }
    bool isButtonTriggered(int b) const { return isButtonPressed(b) && ((m_buttons[1] & (1<<b))==0); }

    void copyToBack()           { m_move[1]=m_move[0]; m_buttons[1]=m_buttons[0]; }
    void setMove(vec2 v)        { m_move[0]=v; }
    void setButtons(int32 v)    { m_buttons[0]=v; }

private:
    vec2 m_move[2];
    int32 m_buttons[2];
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
    vec2 move;
    int32 buttons;
};






} // namespace atomic
#endif // atomic_Game_Input_h
