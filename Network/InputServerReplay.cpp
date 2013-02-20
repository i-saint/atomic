#include "stdafx.h"
#include "InputServerInternal.h"
#include "LevelEditorCommand.h"
#include "Game/AtomicApplication.h"
#include "Game/AtomicGame.h"
#include "Game/World.h"

namespace atomic {


class InputServerReplay
    : public IInputServer
    , public InputServerCommon
{
typedef InputServerCommon impl;
public:
    InputServerReplay();
    virtual IS_TypeID getTypeID() const;

    virtual void update();
    virtual void draw() {}
    virtual bool sync() { return true; }
    virtual void addPlayer(PlayerID pid, const PlayerName &name, uint32 equip) {}
    virtual void erasePlayer(PlayerID pid) {}
    virtual void pushInput(PlayerID pid, const RepInput &is) {}
    virtual void pushLevelEditorCommand(const LevelEditorCommand &v) {}
    virtual void handlePMessage(const PMessage &v) {}
    virtual const InputState& getInput(uint32 pid) const;

    virtual bool save(const char *path) { return false; }
    virtual bool load(const char *path);
    virtual uint32 getPlayLength() const  { return m_header.total_frame; }
    virtual uint32 getPlayPosition() const{ return m_pos; }

private:
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
            m_is[pid].update(rd);
            ++pid;
        }
        else if(m_pos==pl.begin_frame+pl.num_frame) {
            erasePlayer(i);
        }
    }

    {
        LevelEditorCommand s;
        s.frame = m_pos;
        std::pair<LECCont::iterator, LECCont::iterator> lecs
            = std::equal_range(m_lecs.begin(), m_lecs.end(), s, [&](const LevelEditorCommand &a, const LevelEditorCommand &b){ return a.frame<b.frame; });
        for(LECCont::iterator i=lecs.first; i!=lecs.second; ++i) {
            atomicGetGame()->handleLevelEditorCommands(*i);
        }
    }

    ++m_pos;
}

const InputState& InputServerReplay::getInput(uint32 pid) const
{
    return m_is[pid];
}

bool InputServerReplay::load(const char *path)
{
    return impl::load(path);
}

} // namespace atomic
