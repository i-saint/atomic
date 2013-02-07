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
public:
    InputServerNetwork();
    virtual IS_TypeID getTypeID() const;

    virtual void update();
    virtual void addPlayer(uint32 id, const name_t &name, uint32 equip);
    virtual void erasePlayer(uint32 id);
    virtual void pushInput(uint32 pid, const InputState &is);
    virtual void pushLevelEditorCommand(const LevelEditorCommand &v);
    virtual const InputState* getInput() const;

    virtual bool save(const char *path) { return true; }
    virtual bool load(const char *path) { return false; }
    virtual uint32 getPlayLength() const { return 0; }
    virtual uint32 getPlayPosition() const { return 0; }

private:
};

//IInputServer* CreateInputServerLocal() { return istNew(InputServerNetwork)(); }

} // namespace atomic
