#include "stdafx.h"
#include "InputServerInternal.h"
#include "LevelEditorCommand.h"
#include "Game/AtomicApplication.h"
#include "Game/AtomicGame.h"
#include "Game/World.h"

namespace atomic {


bool InputServerCommon::save(const char *path)
{
    ist::GZFileStream gzf(path, "wb");
    if(!gzf.isOpened()) { return false; }

    m_header.random_seed = atomicGetRandom()->getSeed();
    m_header.total_frame = atomicGetFrame();
    m_header.num_players = m_players.size();
    m_header.num_lecs = m_lecs.size();
    gzf.write((char*)&m_header, sizeof(m_header));

    if(!m_players.empty()) {
        gzf.write((char*)&m_players[0], sizeof(RepPlayer)*m_players.size());
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

bool InputServerCommon::load(const char *path)
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




class InputServerLocal
    : public IInputServer
    , public InputServerCommon
{
typedef InputServerCommon impl;
public:
    InputServerLocal();
    virtual IS_TypeID getTypeID() const;

    virtual void update();
    virtual bool sync() { return true; }
    virtual void addPlayer(PlayerID pid, const PlayerName &name, uint32 equip);
    virtual void erasePlayer(PlayerID pid);
    virtual void pushInput(PlayerID pid, const RepInput &is);
    virtual void pushLevelEditorCommand(const LevelEditorCommand &v);
    virtual void handlePMessage(const PMessage &v);
    virtual const InputState& getInput(PlayerID pid) const;

     virtual bool save(const char *path);
     virtual bool load(const char *path) { return false; }
     virtual uint32 getPlayLength() const { return 0; }
     virtual uint32 getPlayPosition() const { return 0; }

private:
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

void InputServerLocal::addPlayer( PlayerID pid, const PlayerName &name, uint32 equip )
{
    RepPlayer t;
    while(m_players.size()<=pid) { m_players.push_back(t); }
    while(m_inputs.size()<=pid) { m_inputs.push_back(InputCont()); }

    wcsncpy(t.name, name, _countof(t.name));
    t.name[_countof(t.name)-1] = '\0';
    t.equip = equip;
    t.begin_frame = atomicGetGame() ? atomicGetFrame() : 0;
    t.num_frame = 0;
    m_players[pid] = t;
}

void InputServerLocal::erasePlayer( PlayerID id )
{
}

void InputServerLocal::pushInput(PlayerID pid, const RepInput &is)
{
    if(pid >= m_players.size()) { istAssert(false); }

    m_inputs[pid].push_back(is);
    m_players[pid].num_frame = m_inputs[pid].size();

    m_is[0].update(is);
}

void InputServerLocal::pushLevelEditorCommand( const LevelEditorCommand &v )
{
    LevelEditorCommand tmp = v;
    tmp.frame = atomicGetFrame();
    atomicGetGame()->handleLevelEditorCommands(tmp);
    m_lecs.push_back(tmp);
}

void InputServerLocal::handlePMessage(const PMessage &v)
{
}


const InputState& InputServerLocal::getInput(PlayerID pid) const
{
    return m_is[pid];
}

bool InputServerLocal::save(const char *path)
{
    return impl::save(path);
}

} // namespace atomic
