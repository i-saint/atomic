#include <windows.h>
#include <mmsystem.h>
#include "HTTPInput.h"

typedef MMRESULT (WINAPI *joyGetPosExT)(UINT uJoyID, LPJOYINFOEX pji);
joyGetPosExT orig_joyGetPosEx;

MMRESULT WINAPI fake_joyGetPosEx(UINT uJoyID, LPJOYINFOEX pji)
{
    StartHTTPInputServer();
    MMRESULT r = orig_joyGetPosEx(uJoyID, pji);
    if(r!=JOYERR_NOERROR) {
        memset(pji, 0, sizeof(JOYINFOEX));
    }
    if(uJoyID<HTTPInputData::MaxPads) {
        const HTTPInputData::Pad &vpad = GetHTTPInputData()->pad[uJoyID];
        pji->dwXpos = abs(vpad.x1+INT16_MIN)>abs((int)pji->dwXpos+INT16_MIN) ? vpad.x1 : pji->dwXpos;
        pji->dwYpos = abs(vpad.y1+INT16_MIN)>abs((int)pji->dwYpos+INT16_MIN) ? vpad.y1 : pji->dwYpos;
        for(int32 i=0; i<HTTPInputData::Pad::MaxButtons; ++i) {
            pji->dwButtons |= vpad.buttons[i]&0x80 ? 1<<i : 0;
        }
    }
    return JOYERR_NOERROR;
}

bool HookMMJoustick()
{
    bool ret = false;
    EachImportFunctionInEveryModule("winmm.dll",
        [&](const char *funcname, void *&func){
            if(strcmp(funcname, "joyGetPosEx")==0) {
                (void*&)orig_joyGetPosEx = func;
                ForceWrite<void*>(func, fake_joyGetPosEx);
                ret = true;
            }
        });
    return ret;
}
