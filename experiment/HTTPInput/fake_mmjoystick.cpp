#include <windows.h>
#include <mmsystem.h>
#include "HTTPInput.h"

typedef MMRESULT (WINAPI *joyGetPosExT)(UINT uJoyID, LPJOYINFOEX pji);
joyGetPosExT orig_joyGetPosEx;

MMRESULT WINAPI fake_joyGetPosEx(UINT uJoyID, LPJOYINFOEX pji)
{
    StartHTTPInputServer();
    MMRESULT r = orig_joyGetPosEx(uJoyID, pji);
    if(r==JOYERR_NOERROR) {
        const HTTPInputData *input = GetHTTPInputData();
        pji->dwXpos = abs(input->pad.x1+INT16_MIN)>abs((int)pji->dwXpos+INT16_MIN) ? input->pad.x1 : pji->dwXpos;
        pji->dwYpos = abs(input->pad.y1+INT16_MIN)>abs((int)pji->dwYpos+INT16_MIN) ? input->pad.y1 : pji->dwYpos;
        for(int i=0; i<32; ++i) {
            pji->dwButtons |= input->pad.buttons & (1<<i);
        }
    }
    return r;
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
