#ifndef atm_Network_InputServerInternal_h
#define atm_Network_InputServerInternal_h

#include "InputServer.h"

namespace atm {

class InputServerCommon
{
public:
    typedef ist::vector<RepPlayer> PlayerCont;
    typedef ist::raw_vector<RepInput> InputCont;
    typedef ist::vector<InputCont> InputConts;
    typedef ist::raw_vector<LevelEditorCommand> LECCont;

protected:
    bool save(const char *path);
    bool load(const char *path);

protected:
    RepHeader m_header;
    PlayerCont m_players;
    InputConts m_inputs;
    LECCont m_lecs;
    InputState m_is[atm_MaxPlayerNum];
};

} // namespace atm
#endif // atm_Network_InputServerInternal_h
