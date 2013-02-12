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
    virtual void handlePMessage(const PMessage &v);
    virtual const InputState& getInput() const;

    virtual bool save(const char *path);
    virtual bool load(const char *path) { return false; }
    virtual uint32 getPlayLength() const { return 0; }
    virtual uint32 getPlayPosition() const { return 0; }

private:
};

InputServerNetwork::InputServerNetwork()
{
}

IInputServer::IS_TypeID InputServerNetwork::getTypeID() const
{
    return IS_Network;
}

void InputServerNetwork::update()
{
}

void InputServerNetwork::addPlayer( uint32 id, const name_t &name, uint32 equip )
{
}

void InputServerNetwork::erasePlayer( uint32 id )
{
}

void InputServerNetwork::pushInput( uint32 pid, const InputState &is )
{
    PMessage_Update mes;
    mes.frame = atomicGetFrame();
    mes.input.move = is.getRawMove();
    mes.input.buttons = is.getButtons();
    atomicGameClientPushMessage(mes);
}

void InputServerNetwork::pushLevelEditorCommand( const LevelEditorCommand &v )
{

}

void InputServerNetwork::handlePMessage( const PMessage &v )
{

}

const InputState& InputServerNetwork::getInput() const
{
    static InputState dummy;
    return dummy;
}

bool InputServerNetwork::save( const char *path )
{
    return false;
}

//IInputServer* CreateInputServerLocal() { return istNew(InputServerNetwork)(); }

} // namespace atomic
