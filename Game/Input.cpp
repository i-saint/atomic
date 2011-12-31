#include "stdafx.h"
#include "features.h"
#include "types.h"
#include "Input.h"
#include "AtomicApplication.h"
#include "AtomicGame.h"

namespace atomic {

namespace  {
    const char magic_string[8] = "atomic\x00";
}

struct RawInputHeader
{
    union {
        struct {
            char magic[8];
            uint32 version;
            uint32 random_seed;
            uint32 num_data;
        };
        uint32 reserved[16];
    };

    void initialize();
    bool isValid();
};

void RawInputHeader::initialize()
{
    memset(this, 0, sizeof(*this));
    memcpy(magic, magic_string, sizeof(magic));
    version = __atomic_replay_version__;
}

bool RawInputHeader::isValid()
{
    if( memcmp(magic, magic_string, sizeof(magic))==0 &&
        version==__atomic_replay_version__)
    {
        return true;
    }
    return false;
}



InputServerLocal::InputServerLocal()
{
    m_data.reserve(60*20); // 20 •ª—\–ñ
}
IInputServer::IS_CLASS InputServerLocal::getClassID() const { return IS_LOCAL; }

void InputServerLocal::update(const InputState &is)
{
    RawInputData rd;
    rd.move = is.getMove();
    rd.buttons = is.getButtons();
    m_data.push_back(rd);

    m_is.copyToBack();
    m_is.setMove(rd.move);
    m_is.setButtons(rd.buttons);
}

const InputState* InputServerLocal::getInput() const
{
    return &m_is;
}

bool InputServerLocal::writeToFile(const char *path)
{
    ist::gzbiostream gzf(path, "wb");
    if(!gzf.isOpened()) { return false; }

    RawInputHeader header;
    header.initialize();
    header.num_data = m_data.size();
    header.random_seed = atomicGetRandom()->getSeed();
    gzf.write((char*)&header, sizeof(header));
    gzf.write((char*)&m_data[0], sizeof(RawInputData)*m_data.size());
    return true;
}


InputServerReplay::InputServerReplay() : m_pos(0) {}
IInputServer::IS_CLASS InputServerReplay::getClassID() const { return IS_REPLAY; }

void InputServerReplay::update(const InputState &is)
{
    if(m_pos < m_data.size()) {
        RawInputData &rd = m_data[m_pos++];
        m_is.copyToBack();
        m_is.setMove(rd.move);
        m_is.setButtons(rd.buttons);
    }
}

const InputState* InputServerReplay::getInput() const
{
    return &m_is;
}

bool InputServerReplay::readFromFile(const char *path)
{
    ist::gzbiostream gzf;
    gzf.open(path, "rb");
    if(!gzf.isOpened()) { return false; }

    RawInputHeader header;
    gzf.read((char*)&header, sizeof(header));
    if(!header.isValid()) { return false; }

    atomicGetRandom()->initialize(header.random_seed);

    m_data.resize(header.num_data);
    gzf.read((char*)&m_data[0], sizeof(RawInputData)*m_data.size());

    return true;
}

} // namespace atomic
