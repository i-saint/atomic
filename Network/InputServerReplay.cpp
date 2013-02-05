#include "stdafx.h"
#include "InputServerInternal.h"
#include "LevelEditorServer.h"
#include "Game/AtomicApplication.h"
#include "Game/AtomicGame.h"
#include "Game/World.h"

namespace atomic {


class InputServerReplay
    : public IInputServer
    , public InputServerCommon
{
public:
    InputServerReplay();
    virtual IS_TypeID getTypeID() const;

    virtual void update();
    virtual void addPlayer(uint32 pid, const char *name, uint32 equip);
    virtual void erasePlayer(uint32 pid);
    virtual void pushInput(uint32 pid, const InputState &is);
    virtual void pushLevelEditorCommand(const LevelEditorCommand &v);
    virtual const InputState* getInput(uint32 pid) const;

    virtual bool save(const char *path) { return false; }
    virtual bool load(const char *path);
    virtual uint32 getPlayLength() const  { return m_header.total_frame; }
    virtual uint32 getPlayPosition() const{ return m_pos; }

private:
    RepHeader m_header;
    PlayerCont m_players;
    InputConts m_inputs;
    LECCont m_lecs;

    InputState m_is[atomic_MaxPlayerNum];
    uint32 m_pos;
};

IInputServer* CreateInputServerReplay() { return istNew(InputServerReplay)(); }


InputServerReplay::InputServerReplay() : m_pos(0) {}
IInputServer::IS_TypeID InputServerReplay::getTypeID() const { return IS_Replay; }

void InputServerReplay::update()
{
    uint32 pid = 0;
    for(size_t i=0; i<m_players.size(); ++i) {
        const RepPlayer &pl = m_players[i];
        if(m_pos>=pl.begin_frame && m_pos<pl.begin_frame+pl.num_frame) {
            InputCont &inp = m_inputs[i];
            RepInput &rd = inp[m_pos-pl.begin_frame];
            m_is[pid].copyToBack();
            m_is[pid].setMove(rd.move);
            m_is[pid].setButtons(rd.buttons);
            ++pid;
        }
        else if(m_pos==pl.begin_frame+pl.num_frame) {
            erasePlayer(i);
        }
    }

    LevelEditorCommand s;
    s.frame = m_pos;
    std::pair<LECCont::iterator, LECCont::iterator> lecs
        = std::equal_range(m_lecs.begin(), m_lecs.end(), s, [&](const LevelEditorCommand &a, const LevelEditorCommand &b){ return a.frame<b.frame; });
    for(LECCont::iterator i=lecs.first; i!=lecs.second; ++i) {
        atomicGetGame()->handleLevelEditorCommands(*i);
    }

    ++m_pos;
}

void InputServerReplay::addPlayer( uint32 id, const char *name, uint32 equip )
{
}

void InputServerReplay::erasePlayer( uint32 id )
{
}

void InputServerReplay::pushInput(uint32 pid, const InputState &is)
{
}

void InputServerReplay::pushLevelEditorCommand( const LevelEditorCommand &v )
{
}

const InputState* InputServerReplay::getInput(uint32 pid) const
{
    return &m_is[pid];
}

bool InputServerReplay::load(const char *path)
{
    ist::GZFileStream gzf;
    gzf.open(path, "rb");
    if(!gzf.isOpened()) { return false; }

    gzf.read((char*)&m_header, sizeof(m_header));
    if(!m_header.isValid()) { return false; }

    atomicGetRandom()->initialize(m_header.random_seed);
    m_players.resize(m_header.num_players);
    m_inputs.resize(m_header.num_players);
    m_lecs.resize(m_header.num_lecs);
    if(!m_players.empty()) {
        gzf.read((char*)&m_players[0], sizeof(RepPlayer)*m_players.size());
    }
    for(size_t i=0; i<m_inputs.size(); ++i) {
        InputCont &inp = m_inputs[i];
        inp.resize(m_players[i].num_frame);
        if(!inp.empty()) {
            gzf.read((char*)&inp[0], sizeof(RepInput)*inp.size());
        }
    }
    if(!m_lecs.empty()) {
        gzf.read((char*)&m_lecs[0], sizeof(LevelEditorCommand)*m_lecs.size());
    }

    return true;
}

} // namespace atomic
