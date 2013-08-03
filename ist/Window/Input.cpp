#include "istPCH.h"
#include "Input.h"
#ifdef ist_env_Windows
#   pragma comment(lib, "dinput8.lib")
#   pragma comment(lib, "dxguid.lib")
#   pragma comment(lib, "xinput.lib")
#   pragma comment(lib, "winmm.lib")
#   include <windows.h>
#   include <mmsystem.h>
#   include <xinput.h>
#   include <dinput.h>
#else //
#endif // todo


namespace ist {


class KeyboardDevice : public IKeyboardDevice
{
public:
    KeyboardDevice( uint32 nth )
        : m_nth(nth)
    {
    }

    void release() { istDelete(this); }
    bool isConnected() const { return true; }
    const KeyboardState& getState() const { return m_state; }

    void update()
    {
        m_state.copyToBack();
#ifdef ist_env_Windows
        ::GetKeyboardState( m_state.getRawKeyState() );
#else // todo
#endif 
    }

private:
    KeyboardState m_state;
    uint32 m_nth;
};

class MouseDevice : public IMouseDevice
{
public:
    MouseDevice( uint32 nth )
        : m_nth(nth)
    {
    }

    void release() { istDelete(this); }
    bool isConnected() const { return true; }
    const MouseState& getState() const { return m_state; }

    void update()
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

private:
    MouseState m_state;
    uint32 m_nth;
};




#ifdef ist_env_Windows

class ControlerDevice_Winmm : public IControlerDevice
{
public:
    ControlerDevice_Winmm( uint32 nth )
        : m_nth(nth)
        , m_connected(false)
    {
    }

    virtual void release() { istDelete(this); }
    virtual bool isConnected() const { return m_connected; }
    virtual const ControllerState& getState() const { return m_state; }

    virtual void update()
    {
        JOYINFOEX joyinfo;
        joyinfo.dwSize = sizeof(JOYINFOEX);
        joyinfo.dwFlags = JOY_RETURNALL;
        MMRESULT result = ::joyGetPosEx(m_nth, &joyinfo);
        m_connected = result==JOYERR_NOERROR;
        if(m_connected) {
            vec2 stick1 = vec2(
                 ist::clamp<int16>((int32)joyinfo.dwXpos+INT16_MIN, INT16_MIN+1, INT16_MAX),
                -ist::clamp<int16>((int32)joyinfo.dwYpos+INT16_MIN, INT16_MIN+1, INT16_MAX)
                ) / (float32)INT16_MAX;
            vec2 stick2 = vec2(
                 ist::clamp<int16>((int32)joyinfo.dwUpos+INT16_MIN, INT16_MIN+1, INT16_MAX),
                -ist::clamp<int16>((int32)joyinfo.dwRpos+INT16_MIN, INT16_MIN+1, INT16_MAX)
                ) / (float32)INT16_MAX;
            uint32 buttons = joyinfo.dwButtons << ControllerState::Button_1;
            // joyinfo.dwPOV: 0 ～ 35900, degree * 100

            m_state.setStick1(stick1);
            m_state.setStick2(stick2);
            m_state.setButtons(buttons);
        }
    }

private:
    ControllerState m_state;
    uint32 m_nth;
    bool m_connected;
};

class ControlerDevice_DirectInput8 : public IControlerDevice
{
public:
    static BOOL CALLBACK _callback( const DIDEVICEINSTANCE* pdidInstance, VOID* pContext )
    {
        ControlerDevice_DirectInput8* pInst = (ControlerDevice_DirectInput8*)pContext;
        if(pInst->m_i!=pInst->m_nth) {
            ++pInst->m_i;
            return DIENUM_CONTINUE;
        }

        HRESULT hr;
        hr = pInst->m_dinput->CreateDevice( pdidInstance->guidInstance, &pInst->m_device, NULL );
        if( FAILED(hr) || pInst->m_device==NULL ) { return DIENUM_CONTINUE; }

        pInst->m_device->SetDataFormat( &c_dfDIJoystick2 );
        pInst->m_device->SetCooperativeLevel( NULL, DISCL_NONEXCLUSIVE|DISCL_BACKGROUND );
        return DIENUM_STOP;
    }

public:
    ControlerDevice_DirectInput8( uint32 nth )
        : m_nth(nth)
        , m_connected(false)
        , m_dinput(NULL)
        , m_device(NULL)
        , m_i(0)
    {
        ::DirectInput8Create( ::GetModuleHandle(NULL), DIRECTINPUT_VERSION, IID_IDirectInput8, (VOID**)&m_dinput, NULL );
        m_dinput->EnumDevices(DI8DEVCLASS_GAMECTRL, &ControlerDevice_DirectInput8::_callback, this, DIEDFL_ATTACHEDONLY);
    }

    ~ControlerDevice_DirectInput8()
    {
        if(m_device) { m_device->Release(); }
        if(m_dinput) { m_dinput->Release(); }
    }

    virtual void release() { istDelete(this); }
    virtual bool isConnected() const { return m_connected; }
    virtual const ControllerState& getState() const { return m_state; }

    virtual void update()
    {
        if(m_device==NULL) { return; }
        HRESULT hr;
        hr = m_device->Poll();
        if(FAILED(hr)) {
            hr = m_device->Acquire();
            while(hr==DIERR_INPUTLOST) {
                hr = m_device->Acquire();
            }
        }

        DIJOYSTATE2 state;
        hr = m_device->GetDeviceState(sizeof(DIJOYSTATE2), &state);
        if(SUCCEEDED(hr)) {
            vec2 stick1 = vec2(
                 ist::clamp<int16>((int32)state.lX+INT16_MIN, INT16_MIN+1, INT16_MAX),
                -ist::clamp<int16>((int32)state.lY+INT16_MIN, INT16_MIN+1, INT16_MAX)
                ) / (float32)INT16_MAX;
            vec2 stick2 = vec2(
                 ist::clamp<int16>((int32)state.lRx+INT16_MIN, INT16_MIN+1, INT16_MAX),
                -ist::clamp<int16>((int32)state.lRy+INT16_MIN, INT16_MIN+1, INT16_MAX)
                ) / (float32)INT16_MAX;
            uint32 buttons = 0;

            for( int i=0; i<16; i++ ) {
                if( state.rgbButtons[i] & 0x80 ) {
                    buttons |= 1<< (ControllerState::Button_1+i);
                }
            }
            m_state.setStick1(stick1);
            m_state.setStick2(stick2);
            m_state.setButtons(buttons);
        }
    }

private:
    ControllerState m_state;
    uint32 m_nth;
    bool m_connected;
    IDirectInput8       *m_dinput;
    IDirectInputDevice8 *m_device;
    uint32 m_i;
};

class ControlerDevice_XInput : public IControlerDevice
{
public:
    ControlerDevice_XInput( uint32 nth )
        : m_nth(nth)
        , m_connected(false)
    {
    }

    virtual void release() { istDelete(this); }
    virtual bool isConnected() const { return m_connected; }
    virtual const ControllerState& getState() const { return m_state; }

    virtual void update()
    {
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
    }

private:
    ControllerState m_state;
    uint32 m_nth;
    bool m_connected;
};

typedef ControlerDevice_DirectInput8 ControlerDevice;
//typedef ControlerDevice_Winmm ControlerDevice;
//typedef ControlerDevice_XInput ControlerDevice;
#endif ist_env_Windows








istAPI IKeyboardDevice* CreateKeyboardDevice(uint32 nth)
{
    return istNew(KeyboardDevice)(nth);
}

istAPI IMouseDevice* CreateMouseDevice(uint32 nth)
{
    return istNew(MouseDevice)(nth);
}

istAPI IControlerDevice* CreateControllerDevice(uint32 nth)
{
    return istNew(ControlerDevice)(nth);
}


} // namespace ist
