#include "istPCH.h"
#include "Input.h"
#ifdef ist_env_Windows
#   pragma comment(lib, "xinput.lib")
#   include <windows.h>
#   include <xinput.h>
#else //
#endif // todo


namespace ist {


class KeyboardDevice : public IKeyboardDevice
{
public:
    KeyboardDevice(uint32 nth);
    virtual ~KeyboardDevice();
    virtual void release();
    virtual void update();
    virtual bool isConnected() const;
    virtual const KeyboardState& getState() const;

private:
    KeyboardState m_state;
    uint32 m_nth;
};

class MouseDevice : public IMouseDevice
{
public:
    MouseDevice(uint32 nth);
    virtual ~MouseDevice();
    virtual void release();
    virtual void update();
    virtual bool isConnected() const;
    virtual const MouseState& getState() const;

private:
    MouseState m_state;
    uint32 m_nth;
};

class ControlerDevice : public IControlerDevice
{
public:
    ControlerDevice(uint32 nth);
    virtual ~ControlerDevice();
    virtual void release();
    virtual void update();
    virtual bool isConnected() const;
    virtual const ControllerState& getState() const;

private:
    ControllerState m_state;
    uint32 m_nth;
    bool m_connected;
};




KeyboardDevice::KeyboardDevice( uint32 nth )
    : m_nth(nth)
{
}

KeyboardDevice::~KeyboardDevice()
{
}

void KeyboardDevice::release()
{
    istDelete(this);
}

void KeyboardDevice::update()
{
    m_state.copyToBack();
#ifdef ist_env_Windows
    ::GetKeyboardState( m_state.getRawKeyState() );
#else // todo
#endif 
}

bool KeyboardDevice::isConnected() const
{
    return true;
}

const KeyboardState& KeyboardDevice::getState() const
{
    return m_state;
}



MouseDevice::MouseDevice( uint32 nth )
    : m_nth(nth)
{
}

MouseDevice::~MouseDevice()
{
}

void MouseDevice::release()
{
    istDelete(this);
}

void MouseDevice::update()
{
#ifdef ist_env_Windows
    {
        CURSORINFO cinfo;
        cinfo.cbSize = sizeof(cinfo);
        ::GetCursorInfo(&cinfo);
        m_state.setPosition( ivec2(cinfo.ptScreenPos.x, cinfo.ptScreenPos.y) );
    }
    {
        uint32 buttons = 0;
        uint8 key[256];
        ::GetKeyboardState(key);
        buttons |= (key[VK_LBUTTON]&0x80) ? MouseState::Button_Left : 0;
        buttons |= (key[VK_RBUTTON]&0x80) ? MouseState::Button_Right : 0;
        buttons |= (key[VK_MBUTTON]&0x80) ? MouseState::Button_Middle : 0;
        m_state.setButtons(buttons);
    }
#else // todo
#endif 
}

bool MouseDevice::isConnected() const
{
    return true;
}

const MouseState& MouseDevice::getState() const
{
    return m_state;
}



ControlerDevice::ControlerDevice( uint32 nth )
    : m_nth(nth)
    , m_connected(false)
{
}

ControlerDevice::~ControlerDevice()
{
}

void ControlerDevice::release()
{
    istDelete(this);
}

void ControlerDevice::update()
{
#ifdef ist_env_Windows
    XINPUT_STATE state;
    DWORD dwResult = ::XInputGetState( m_nth, &state );
    m_connected = dwResult==ERROR_SUCCESS;
    if(dwResult==ERROR_SUCCESS) {
        vec2 stick1 = vec2(
            ist::clamp<int16>(state.Gamepad.sThumbLX, INT16_MIN+1, INT16_MAX),
            ist::clamp<int16>(state.Gamepad.sThumbLY, INT16_MIN+1, INT16_MAX) ) / (float32)INT16_MAX;
        vec2 stick2 = vec2(
            ist::clamp<int16>(state.Gamepad.sThumbRX, INT16_MIN+1, INT16_MAX),
            ist::clamp<int16>(state.Gamepad.sThumbRY, INT16_MIN+1, INT16_MAX) ) / (float32)INT16_MAX;
        float32 trigger1 = (float32)state.Gamepad.bLeftTrigger / 255.0f;
        float32 trigger2 = (float32)state.Gamepad.bRightTrigger / 255.0f;
        uint32 buttons = state.Gamepad.wButtons;

        m_state.setStick1(stick1);
        m_state.setStick2(stick2);
        m_state.setTrigger1(trigger1);
        m_state.setTrigger2(trigger2);
        m_state.setButtons(buttons);
    }
#endif 
}

bool ControlerDevice::isConnected() const
{
    return m_connected;
}

const ControllerState& ControlerDevice::getState() const
{
    return m_state;
}




istInterModule IKeyboardDevice* CreateKeyboardDevice(uint32 nth)
{
    return istNew(KeyboardDevice)(nth);
}

istInterModule IMouseDevice* CreateMouseDevice(uint32 nth)
{
    return istNew(MouseDevice)(nth);
}

istInterModule IControlerDevice* CreateControllerDevice(uint32 nth)
{
    return istNew(ControlerDevice)(nth);
}


} // namespace ist
