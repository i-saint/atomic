#ifndef ist_Application_InputState_h
#define ist_Application_InputState_h

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
};


class istInterModule KeyboardState
{
private:
    unsigned char   m_keystate[2][256];

public:
    KeyboardState()     { istMemset(this, 0, sizeof(*this)); }
    void copyToBack()   { istMemcpy(m_keystate[1], m_keystate[0], 256); }
    unsigned char* getRawKeyState() { return m_keystate[0]; }
    bool isKeyPressed(int v) const { return (m_keystate[0][v] & 0x80)!=0; }
    bool isKeyTriggered(int v) const { return (m_keystate[0][v] & 0x80)!=0 && (m_keystate[1][v] & 0x80)==0; }
};

class istInterModule MouseState
{
public:
    enum BUTTON
    {
        BU_LEFT     = 0x01,
        BU_RIGHT    = 0x02,
        BU_MIDDLE   = 0x04,
    };

private:
    int m_button;
    int m_x;
    int m_y;

public:
    MouseState() { istMemset(this, 0, sizeof(*this)); }

    int getButtonState() const { return m_button; }
    int getX() const { return m_x; }
    int getY() const { return m_y; }

    void setButtonState(int v) { m_button=v; }
    void setX(int v) { m_x=v; }
    void setY(int v) { m_y=v; }
};

class istInterModule JoyState
{
private:
    int16 m_x;
    int16 m_y;
    int16 m_z;
    int16 m_r;
    int16 m_u;
    int16 m_v;
    int32 m_pov;
    uint32 m_buttons;

public:
    JoyState() { istMemset(this, 0, sizeof(*this)); }
    int32 getX() const { return m_x; } // -32767 ～ 32767
    int32 getY() const { return m_y; } // -32767 ～ 32767
    int32 getZ() const { return m_z; } // -32767 ～ 32767
    int32 getR() const { return m_r; } // -32767 ～ 32767
    int32 getU() const { return m_u; } // -32767 ～ 32767
    int32 getV() const { return m_v; } // -32767 ～ 32767
    int32 getRoV() const { return m_pov; } // 0 ～ 35900, degree * 100
    uint32 getButtons() const { return m_buttons; }
    bool isButtonPressed(int i) const { return (m_buttons & (1<<i))!=0; }

#ifdef ist_env_Windows
    void setValue(const JOYINFOEX& v)
    {
        m_x = ist::clamp<int16>((int32)v.dwXpos - 32768, INT16_MIN+1, INT16_MAX);
        m_y = ist::clamp<int16>((int32)v.dwYpos - 32768, INT16_MIN+1, INT16_MAX);
        m_z = ist::clamp<int16>((int32)v.dwZpos - 32768, INT16_MIN+1, INT16_MAX);
        m_r = ist::clamp<int16>((int32)v.dwRpos - 32768, INT16_MIN+1, INT16_MAX);
        m_u = ist::clamp<int16>((int32)v.dwUpos - 32768, INT16_MIN+1, INT16_MAX);
        m_v = ist::clamp<int16>((int32)v.dwVpos - 32768, INT16_MIN+1, INT16_MAX);
        m_pov = v.dwPOV;
        m_buttons = v.dwButtons;
    }
#endif // ist_env_Windows
};


struct istInterModule DisplaySetting
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
#endif // ist_Application_InputState_h
