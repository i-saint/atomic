#include "stdafx.h"
#include "InputServerInternal.h"
#include "LevelEditorCommand.h"
#include "GameClient.h"
#include "Game/AtomicApplication.h"
#include "Game/AtomicGame.h"
#include "Game/World.h"

namespace atomic {

class InputServerNetwork
    : public IInputServer
    , public InputServerCommon
{
typedef InputServerCommon impl;
public:
    InputServerNetwork();
    virtual IS_TypeID getTypeID() const;

    virtual void update();
    virtual void draw() {}
    virtual bool sync();
    virtual void addPlayer(PlayerID id, const PlayerName &name, uint32 equip);
    virtual void erasePlayer(PlayerID id);
    virtual void pushInput(PlayerID pid, const RepInput &is);
    virtual void pushLevelEditorCommand(const LevelEditorCommand &v);
    virtual void handlePMessage(const PMessage &v);
    virtual const InputState& getInput(PlayerID pid) const;

    virtual bool save(const char *path);
    virtual bool load(const char *path) { return false; }
    virtual uint32 getPlayLength() const { return 0; }
    virtual uint32 getPlayPosition() const { return 0; }

private:
    uint32 m_server_frame;
    uint32 m_pos;
};

InputServerNetwork::InputServerNetwork()
    : m_server_frame(0)
    , m_pos(0)
{
}

IInputServer::IS_TypeID InputServerNetwork::getTypeID() const
{
    return IS_Network;
}

void InputServerNetwork::update()
{
    uint32 pid = 0;
    for(size_t i=0; i<m_players.size(); ++i) {
        const RepPlayer &pl = m_players[i];
        if(m_pos>=pl.begin_frame && pl.num_frame==0) {
            InputCont &inp = m_inputs[i];
            RepInput &rd = inp[m_pos-pl.begin_frame];
            m_is[pid].update(rd);
            ++pid;
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

bool InputServerNetwork::sync()
{
    return m_pos < m_server_frame;
}

void InputServerNetwork::addPlayer( PlayerID id, const PlayerName &name, uint32 equip )
{
}

void InputServerNetwork::erasePlayer( PlayerID id )
{
}

void InputServerNetwork::pushInput( PlayerID pid, const RepInput &is )
{
    atomicGameClientPushMessage( PMessage_Update::create(pid, atomicGetFrame(), is) );
}

void InputServerNetwork::pushLevelEditorCommand( const LevelEditorCommand &v )
{
    atomicGameClientPushMessage( PMessage_LEC::create(v) );
}

void InputServerNetwork::handlePMessage( const PMessage &mes )
{
    switch(mes.type) {
    case PM_Join:
        {
            auto &m = reinterpret_cast<const PMessage_Join&>(mes);
            istAssert(m.player_id<=m_players.size() || m.player_id-m_players.size() < 4);
            while(m_players.size()<=m.player_id) { m_players.push_back(RepPlayer()); }
            while(m_inputs.size()<=m.player_id) { m_inputs.push_back(InputCont()); }

            RepPlayer &t = m_players[m.player_id];
            wcsncpy(t.name, m.name, _countof(t.name));
            t.name[_countof(t.name)-1] = L'\0';
            //t.equip = m.equip;
            t.begin_frame = atomicGetGame() ? atomicGetFrame() : 0;
            t.num_frame = 0;
        }
        break;
    case PM_Update:
        {
            auto &m = reinterpret_cast<const PMessage_Update&>(mes);
            istAssert(m.player_id < m_inputs.size());
            InputCont &input = m_inputs[m.player_id];
            if(input.size() <= m.frame) {
                input.resize(m.frame+1, RepInput());
                input[m.frame] = m.input;
            }
            m_server_frame = std::max<uint32>(m_server_frame, m.server_frame);
        }
        break;
    case PM_LevelEditorCommand:
        {
            auto &m = reinterpret_cast<const PMessage_LEC&>(mes);
            m_lecs.insert(std::lower_bound(m_lecs.begin(), m_lecs.end(), m.lec), m.lec);
        }
        break;
    }
}

const InputState& InputServerNetwork::getInput(PlayerID pid) const
{
    return m_is[pid];
}

bool InputServerNetwork::save( const char *path )
{
    return impl::save(path);
}

IInputServer* CreateInputServerNetwork() { return istNew(InputServerNetwork)(); }

} // namespace atomic
