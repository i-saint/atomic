#ifndef __ist_Application_InputState_h__
#define __ist_Application_InputState_h__
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
    KeyboardState()     { memset(this, 0, sizeof(*this)); }
    void copyToBack()   { memcpy(m_keystate[1], m_keystate[0], 256); }
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
    MouseState() { memset(this, 0, sizeof(*this)); }

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
    int m_x;
    int m_y;
    int m_z;
    int m_r;
    int m_u;
    int m_v;
    int m_pov;
    int m_buttons;

public:
    JoyState() { memset(this, 0, sizeof(*this)); }
    int getX() const { return m_x; } // -32768 ～ 32767
    int getY() const { return m_y; } // -32768 ～ 32767
    int getZ() const { return m_z; } // -32768 ～ 32767
    int getR() const { return m_r; } // -32768 ～ 32767
    int getU() const { return m_u; } // -32768 ～ 32767
    int getV() const { return m_v; } // -32768 ～ 32767
    int getRoV() const { return m_pov; } // 0 ～ 35900, degree * 100
    int getButtons() const { return m_buttons; }
    bool isButtonPressed(int i) const { return (m_buttons & (1<<i))!=0; }

    void setValue(const JOYINFOEX& v)
    {
        m_x = v.dwXpos; m_x -= 32768;
        m_y = v.dwYpos; m_y -= 32768;
        m_z = v.dwZpos; m_z -= 32768;
        m_r = v.dwRpos; m_r -= 32768;
        m_u = v.dwUpos; m_u -= 32768;
        m_v = v.dwVpos; m_v -= 32768;
        m_pov = v.dwPOV;
        m_buttons = v.dwButtons;
    }
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
#endif // __ist_Application_InputState_h__
