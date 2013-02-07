#ifndef atomic_Network_InputServer_h
#define atomic_Network_InputServer_h

#include "Game/Input.h"

namespace atomic {

class IInputServer
{
public:
    typedef wchar_t (name_t)[16];
    enum IS_TypeID {
        IS_Local,
        IS_Replay,
        IS_Network,
    };

    virtual ~IInputServer() {}
    virtual IS_TypeID getTypeID() const=0;

    virtual void update()=0;
    virtual void addPlayer(uint32 pid, const name_t &name, uint32 equip)=0;
    virtual void erasePlayer(uint32 pid)=0;
    virtual void pushInput(uint32 pid, const InputState &v)=0;
    virtual void pushLevelEditorCommand(const LevelEditorCommand &v)=0;
    virtual const InputState* getInput(uint32 pid) const=0;

    virtual bool save(const char *path)=0;
    virtual bool load(const char *path)=0;
    virtual uint32 getPlayLength() const=0;
    virtual uint32 getPlayPosition() const=0;
};

IInputServer* CreateInputServerLocal();
IInputServer* CreateInputServerReplay();
IInputServer* CreateInputServerNetwork();

} // namespace atomic
#endif // atomic_Network_InputServer_h
