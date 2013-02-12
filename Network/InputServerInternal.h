#ifndef atomic_Network_InputServerInternal_h
#define atomic_Network_InputServerInternal_h

#include "InputServer.h"

namespace atomic {

class InputServerCommon
{
public:
    typedef ist::vector<RepPlayer> PlayerCont;
    typedef ist::vector<RepInput> InputCont;
    typedef ist::vector<InputCont> InputConts;
    typedef ist::vector<LevelEditorCommand> LECCont;

    bool save(const char *path);
};

} // namespace atomic
#endif // atomic_Network_InputServerInternal_h
