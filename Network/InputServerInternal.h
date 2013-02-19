#ifndef atomic_Network_InputServerInternal_h
#define atomic_Network_InputServerInternal_h

#include "InputServer.h"

namespace atomic {

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
    InputState m_is[atomic_MaxPlayerNum];
};

} // namespace atomic
#endif // atomic_Network_InputServerInternal_h
