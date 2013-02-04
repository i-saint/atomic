#include "stdafx.h"
#include "features.h"
#include "types.h"
#include "Input.h"
#include "AtomicApplication.h"
#include "AtomicGame.h"
#include "World.h"
#include "Network/LevelEditorServer.h"

namespace atomic {

namespace  {
    const char magic_string[8] = "atomic\x00";
}

RepHeader::RepHeader()
{
    memset(this, 0, sizeof(*this));
    memcpy(magic, magic_string, sizeof(magic));
    version = atomic_replay_version;
}

bool RepHeader::isValid()
{
    if( memcmp(magic, magic_string, sizeof(magic))==0 && version==atomic_replay_version)
    {
        return true;
    }
    return false;
}

RepPlayer::RepPlayer() { memset(this, 0, sizeof(*this)); }



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

bool InputServerLocal::writeToFile(const char *path)
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

bool InputServerReplay::readFromFile(const char *path)
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
