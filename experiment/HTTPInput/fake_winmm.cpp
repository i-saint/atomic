#include <windows.h>
#include <mmsystem.h>
#include "HTTPInput.h"
#include "HTTPInput_Internal.h"

typedef MMRESULT (WINAPI *joyGetPosExT)(UINT uJoyID, LPJOYINFOEX pji);
static joyGetPosExT orig_joyGetPosEx;

static MMRESULT WINAPI fake_joyGetPosEx(UINT uJoyID, LPJOYINFOEX pji)
{
    MMRESULT r = orig_joyGetPosEx(uJoyID, pji);
    if(HTTPInput_GetConfig()->override_winmm) {
        HTTPInput_StartServer();
        if(r!=JOYERR_NOERROR) { memset(pji, 0, sizeof(JOYINFOEX)); }
        if(uJoyID<HTTPInputData::MaxPads) {
            const HTTPInputData::Pad &vpad = HTTPInput_GetData()->pad[uJoyID];
            pji->dwXpos = abs(vpad.x1+INT16_MIN)>abs((int)pji->dwXpos+INT16_MIN) ? vpad.x1 : pji->dwXpos;
            pji->dwYpos = abs(vpad.y1+INT16_MIN)>abs((int)pji->dwYpos+INT16_MIN) ? vpad.y1 : pji->dwYpos;
            LONG trigger = vpad.trigger1 > vpad.trigger2 ? -vpad.trigger1 : vpad.trigger2;
            pji->dwZpos = abs(trigger)>0x100 ? INT16_MAX+trigger : pji->dwYpos;
            for(int32 i=0; i<HTTPInputData::Pad::MaxButtons; ++i) {
                pji->dwButtons |= vpad.buttons[i]&0x80 ? 1<<i : 0;
            }
        }
        return JOYERR_NOERROR;
    }
    return r;
}

static FuncInfo g_winmm_funcs[] = {
    {"joyGetPosEx", 0, (void*)&fake_joyGetPosEx, (void**)&orig_joyGetPosEx},
};
OverrideInfo g_winmm_overrides = {"winmm.dll", _countof(g_winmm_funcs), g_winmm_funcs};
