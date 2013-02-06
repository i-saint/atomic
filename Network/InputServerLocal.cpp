#include "stdafx.h"
#include "InputServerInternal.h"
#include "LevelEditorCommand.h"
#include "Game/AtomicApplication.h"
#include "Game/AtomicGame.h"
#include "Game/World.h"

namespace atomic {

class InputServerLocal
    : public IInputServer
    , public InputServerCommon
{
public:
    InputServerLocal();
    virtual IS_TypeID getTypeID() const;

    virtual void update();
    virtual void addPlayer(uint32 pid, const char *name, uint32 equip);
    virtual void erasePlayer(uint32 pid);
    virtual void pushInput(uint32 pid, const InputState &is);
    virtual void pushLevelEditorCommand(const LevelEditorCommand &v);
    virtual const InputState* getInput(uint32 pid) const;

     virtual bool save(const char *path);
     virtual bool load(const char *path) { return false; }
     virtual uint32 getPlayLength() const { return 0; }
     virtual uint32 getPlayPosition() const { return 0; }

private:
    PlayerCont m_playes;
    InputConts m_inputs;
    LECCont m_lecs;

    InputState m_is;
};

IInputServer* CreateInputServerLocal() { return istNew(InputServerLocal)(); }


InputServerLocal::InputServerLocal()
{
    m_inputs.reserve(60*20); // 20 分予約
}
IInputServer::IS_TypeID InputServerLocal::getTypeID() const { return IS_Local; }

void InputServerLocal::update()
{
}

void InputServerLocal::addPlayer( uint32 pid, const char *name, uint32 equip )
{
    RepPlayer t;
    while(m_playes.size()<=pid) { m_playes.push_back(t); }
    while(m_inputs.size()<=pid) { m_inputs.push_back(InputCont()); }

    strncpy(t.name, name, _countof(t.name));
    t.name[_countof(t.name)-1] = '\0';
    t.equip = equip;
    t.begin_frame = atomicGetGame() ? atomicGetFrame() : 0;
    t.num_frame = 0;
    m_playes[pid] = t;
}

void InputServerLocal::erasePlayer( uint32 id )
{
}

void InputServerLocal::pushInput(uint32 pid, const InputState &is)
{
    if(pid >= m_playes.size()) { istAssert(false); }

    RepInput rd;
    rd.move = is.getMove();
    rd.buttons = is.getButtons();
    m_inputs[pid].push_back(rd);
    m_playes[pid].num_frame = m_inputs[pid].size();

    m_is.copyToBack();
    m_is.setMove(rd.move);
    m_is.setButtons(rd.buttons);
}

void InputServerLocal::pushLevelEditorCommand( const LevelEditorCommand &v )
{
    LevelEditorCommand tmp = v;
    tmp.frame = atomicGetFrame();
    atomicGetGame()->handleLevelEditorCommands(tmp);
    m_lecs.push_back(tmp);
}

const InputState* InputServerLocal::getInput(uint32 pid) const
{
    return &m_is;
}

bool InputServerLocal::save(const char *path)
{
    ist::GZFileStream gzf(path, "wb");
    if(!gzf.isOpened()) { return false; }

    RepHeader header;
    header.random_seed = atomicGetRandom()->getSeed();
    header.total_frame = atomicGetFrame();
    header.num_players = m_playes.size();
    header.num_lecs = m_lecs.size();
    gzf.write((char*)&header, sizeof(header));

    if(!m_playes.empty()) {
        gzf.write((char*)&m_playes[0], sizeof(RepPlayer)*m_playes.size());
    }
    for(size_t i=0; i<m_inputs.size(); ++i) {
        InputCont &inp = m_inputs[i];
        if(!inp.empty()) {
            gzf.write((char*)&inp[0], sizeof(RepInput)*inp.size());
        }
    }
    if(!m_lecs.empty()) {
        gzf.write((char*)&m_lecs[0], sizeof(LevelEditorCommand)*m_lecs.size());
    }
    return true;
}

} // namespace atomic
