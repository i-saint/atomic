#ifndef __atomic_Game_Input__
#define __atomic_Game_Input__
namespace atomic {

class InputState
{
private:
    vec2 m_move;
    int32 m_buttons[2];

public:
    InputState()
    {
        m_move = vec2(0.0f);
        stl::fill_n(m_buttons, _countof(m_buttons), 0);
    }

    vec2 getMove() const                { return m_move; }
    bool isButtonPressed(int b) const   { return (m_buttons[0] & (1<<b)) !=0; }
    bool isButtonTriggered(int b) const { return isButtonPressed(b) && ((m_buttons[1] & (1<<b))==0); }
    int32 getButtons() const            { return m_buttons[0]; }

    void copyToBack()           { m_buttons[1]=m_buttons[0]; }
    void setMove(vec2 v)        { m_move=v; }
    void setButtons(int32 v)    { m_buttons[0]=v; }
};


class IInputServer
{
public:
    enum IS_CLASS {
        IS_LOCAL,
        IS_REPLAY,
        IS_NET_SERVER,
        IS_NET_CLIENT,
    };

    virtual ~IInputServer() {}
    virtual IS_CLASS getClassID() const=0;
    virtual void update(const InputState &is)=0;
    virtual const InputState* getInput() const=0;
};



struct RawInputData
{
    vec2 move;
    int32 buttons;
};


class InputServerLocal : public IInputServer
{
private:
    stl::vector<RawInputData> m_data;
    InputState m_is;

public:
    InputServerLocal();
    IS_CLASS getClassID() const;
    void update(const InputState &is);
    const InputState* getInput() const;

    bool writeToFile(const char *path);
};

class InputServerReplay : public IInputServer
{
private:
    stl::vector<RawInputData> m_data;
    InputState m_is;
    uint32 m_pos;

public:
    InputServerReplay();
    IS_CLASS getClassID() const;
    void update(const InputState &is);
    const InputState* getInput() const;

    bool readFromFile(const char *path);
    uint32 getReplayLength() const  { return m_data.size(); }
    uint32 getReplayPosition() const{ return m_pos; }
};


} // namespace atomic
#endif // __atomic_Game_Input__
