#include <windows.h>
#include <xinput.h>
#include "HTTPInput.h"

typedef DWORD (WINAPI *XInputGetStateT)(DWORD dwUserIndex, XINPUT_STATE* pState);
XInputGetStateT orig_XInputGetState;

DWORD WINAPI fake_XInputGetState(DWORD dwUserIndex, XINPUT_STATE* pState)
{
    StartHTTPInputServer();
    DWORD r = orig_XInputGetState(dwUserIndex, pState);
    if(r!=ERROR_SUCCESS) {
        memset(pState, 0, sizeof(XINPUT_STATE));
    }
    if(dwUserIndex<HTTPInputData::MaxPads) {
        XINPUT_GAMEPAD &pad = pState->Gamepad;
        const HTTPInputData::Pad &vpad = GetHTTPInputData()->pad[dwUserIndex];
        pad.sThumbLX = abs(vpad.x1+INT16_MIN)>abs((int)pad.sThumbLX) ?  vpad.x1+INT16_MIN : pad.sThumbLX;
        pad.sThumbLY = abs(vpad.y1+INT16_MIN)>abs((int)pad.sThumbLY) ?-(vpad.y1-INT16_MAX): pad.sThumbLY;
        for(int32 i=0; i<4; ++i) {
            pad.wButtons |= (vpad.buttons[i] & 0x80 ? 1 : 0)<<12;
        }
    }
    return ERROR_SUCCESS;
}

bool HookXInput()
{
    bool ret = false;
    EachImportFunctionInEveryModule("xinput.+\\.dll",
        [](const char *funcname, void *&func) {},
        [&](DWORD ordinal, void *&func){
            if(ordinal==2) {
                (void*&)orig_XInputGetState = func;
                ForceWrite<void*>(func, fake_XInputGetState);
                ret = true;
            }
        });
    return ret;
}
