#ifndef ist_Application_Input_h
#define ist_Application_Input_h

#include "ist/Math/Misc.h"

namespace ist {

enum KEY {
    KEY_RIGHT   = VK_RIGHT,
    KEY_LEFT    = VK_LEFT,
    KEY_UP      = VK_UP,
    KEY_DOWN    = VK_DOWN,
    KEY_ESCAPE  = VK_ESCAPE,
    KEY_F1      = VK_F1,
    KEY_F2      = VK_F2,
    KEY_F3      = VK_F3,
    KEY_F4      = VK_F4,
    KEY_F5      = VK_F5,
    KEY_F6      = VK_F6,
    KEY_F7      = VK_F7,
    KEY_F8      = VK_F8,
    KEY_F9      = VK_F9,
    KEY_F10     = VK_F10,
    KEY_F11     = VK_F11,
    KEY_DELETE  = VK_DELETE,
    KEY_BACK    = VK_BACK,
};


class KeyboardState
{
public:
    KeyboardState()     { istMemset(this, 0, sizeof(*this)); }
    void copyToBack()   { istMemcpy(m_keystate[1], m_keystate[0], 256); }
    uint8* getRawKeyState() { return m_keystate[0]; }
    bool isKeyPressed(uint32 v) const { return (m_keystate[0][v] & 0x80)!=0; }
    bool isKeyTriggered(uint32 v) const { return (m_keystate[0][v] & 0x80)!=0 && (m_keystate[1][v] & 0x80)==0; }

private:
    uint8 m_keystate[2][256];
};

class MouseState
{
public:
    enum Button
    {
        Button_Left,
        Button_Right,
        Button_Middle,
    };

public:
    MouseState() : m_buttons(0) {}

    uint32       getButtons() const { return m_buttons; }
    const ivec2& getPosition() const{ return m_pos; }
    bool isButtonPressed(Button i) const { return (m_buttons & (1<<i))!=0; }

    void setButtons(uint32 v)       { m_buttons=v; }
    void setPosition(const ivec2& v){ m_pos; }

private:
    ivec2 m_pos;
    uint32 m_buttons;
};

class ControllerState
{
public:
    enum Button
    {
        DPad_Up,
        DPad_Down,
        DPad_Left,
        DPad_Right,
        Button_Start,
        Button_Back,
        Button_LThumb,
        Button_RThumb,
        Button_LShoulder,
        Button_RShoulder,
        Button_1 = 12,
        Button_2,
        Button_3,
        Button_4,
        Button_5,
        Button_6,
        Button_7,
        Button_8,
    };

public:
    ControllerState() : m_trigger1(0.0f), m_trigger2(0.0f), m_buttons(0)
    {}

    const vec2& getStick1() const   { return m_stick1; }
    const vec2& getStick2() const   { return m_stick2; }
    float32     getTrigger1() const { return m_trigger1; }
    float32     getTrigger2() const { return m_trigger2; }
    uint32      getButtons() const  { return m_buttons; }
    bool        isButtonPressed(Button i) const { return (m_buttons & (1<<i))!=0; }

    void setStick1(const vec2 &v)   { m_stick1=v; }
    void setStick2(const vec2 &v)   { m_stick2=v; }
    void setTrigger1(float32 v)     { m_trigger1=v; }
    void setTrigger2(float32 v)     { m_trigger2=v; }
    void setButtons(uint32 v)       { m_buttons=v; }

private:
    vec2    m_stick1;
    vec2    m_stick2;
    float32 m_trigger1;
    float32 m_trigger2;
    uint32  m_buttons;
};


class istInterModule IKeyboardDevice
{
protected:
    virtual ~IKeyboardDevice() {}
public:
    virtual void release()=0;
    virtual void update()=0;
    virtual bool isConnected() const=0;
    virtual const KeyboardState& getState() const=0;
};

class istInterModule IMouseDevice
{
protected:
    virtual ~IMouseDevice() {}
public:
    virtual void release()=0;
    virtual void update()=0;
    virtual bool isConnected() const=0;
    virtual const MouseState& getState() const=0;
};

class istInterModule IControlerDevice
{
protected:
    virtual ~IControlerDevice() {}
public:
    virtual void release()=0;
    virtual void update()=0;
    virtual bool isConnected() const=0;
    virtual const ControllerState& getState() const=0;
};

istInterModule IKeyboardDevice*    CreateKeyboardDevice(uint32 nth=0);
istInterModule IMouseDevice*       CreateMouseDevice(uint32 nth=0);
istInterModule IControlerDevice*   CreateControllerDevice(uint32 nth=0);




struct DisplaySetting
{
private:
    ivec2 m_resolution;
    int m_color_bits;
    int m_frequency;

public:
    DisplaySetting() : m_resolution(0,0), m_color_bits(0), m_frequency(0) {}
    DisplaySetting(ivec2 res, int bits, int freq) : m_resolution(res), m_color_bits(bits), m_frequency(freq) {}

    ivec2 getResolution() const { return m_resolution; }
    int getColorBits() const    { return m_color_bits; }
    int getFrequency() const    { return m_frequency; }
};

} // namspace ist
#endif // ist_Application_Input_h
