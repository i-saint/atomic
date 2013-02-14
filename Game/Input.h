#ifndef atomic_Game_Input_h
#define atomic_Game_Input_h

#include "types.h"

namespace atomic {

union LevelEditorCommand;


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
    PlayerName name;
    uint32 equip;
    uint32 begin_frame;
    uint32 num_frame;

    RepPlayer();
};

struct istAlign(4) RepMove
{
    int16 x,y;

    RepMove() : x(0),y(0) {}
    RepMove(int16 _x, int16 _y) : x(_x) , y(_y) {}
    vec2 toF() const { return vec2(float32(x),float32(y))/32767.0f; }
};
typedef uint32 RepButton;

struct RepInput
{
    RepMove move;
    RepButton buttons;

    RepInput() : move(0,0), buttons(0) {}
    RepInput(RepMove m, RepButton b) : move(m), buttons(b) {}
};


class InputState
{
public:
    enum Direction {
        Dir_Left,
        Dir_Right,
        Dir_Up,
        Dir_Down,
    };

    InputState()
    {
        stl::fill_n(m_in, _countof(m_in), RepInput(RepMove(0,0), RepButton(0)));
    }

    const RepInput& getRawInput() const { return m_in[0]; }
    vec2 getMove() const { return m_in[0].move.toF(); }
    bool isDirectionPressed(Direction d) const
    {
        switch(d) {
        case Dir_Right: return m_in[0].move.x>= 0.5f;
        case Dir_Left:  return m_in[0].move.x<=-0.5f;
        case Dir_Up:    return m_in[0].move.y>= 0.5f;
        case Dir_Down:  return m_in[0].move.y<=-0.5f;
        }
        return false;
    }
    bool isDirectionTriggered(Direction d) const
    {
        return isDirectionPressed(d) && (std::abs(m_in[1].move.x)<0.5f && std::abs(m_in[1].move.y)<0.5f);
    }

    int32 getButtons() const            { return m_in[0].buttons; }
    bool isButtonPressed(int b) const   { return (m_in[0].buttons & (1<<b)) !=0; }
    bool isButtonTriggered(int b) const { return isButtonPressed(b) && ((m_in[1].buttons & (1<<b))==0); }

    void update(const RepInput &inp)
    {
        m_in[1] = m_in[0];
        m_in[0] = inp;
    }

private:
    RepInput m_in[2];
};






} // namespace atomic
#endif // atomic_Game_Input_h
