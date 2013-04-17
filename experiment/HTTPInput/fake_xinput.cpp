#include <windows.h>
#include <xinput.h>
#include "HTTPInput.h"

typedef DWORD (WINAPI *XInputGetStateT)(DWORD dwUserIndex, XINPUT_STATE* pState);
XInputGetStateT orig_XInputGetState;

DWORD WINAPI fake_XInputGetState(DWORD dwUserIndex, XINPUT_STATE* pState)
{
    StartHTTPInputServer();
    DWORD r = orig_XInputGetState(dwUserIndex, pState);
    if(r==ERROR_SUCCESS) {
        XINPUT_GAMEPAD &pad = pState->Gamepad;
        const HTTPInputData *input = GetHTTPInputData();
        pad.sThumbLX = abs(input->pad.x1+INT16_MIN)>abs((int)pad.sThumbLX) ?  input->pad.x1+INT16_MIN : pad.sThumbLX;
        pad.sThumbLY = abs(input->pad.y1+INT16_MIN)>abs((int)pad.sThumbLY) ?-(input->pad.y1-INT16_MAX): pad.sThumbLY;
        for(int i=0; i<4; ++i) {
            pad.wButtons |= (input->pad.buttons & 1<<i)<<12;
        }
    }
    return r;
}

bool HookXInput()
{
    bool ret = false;
    EachImportFunctionInEveryModule(
        [](const char *dllname) {
            return _strnicmp(dllname, "xinput", 6)==0;
        },
        [](const char *funcname, void *&func) {},
        [&](DWORD ordinal, void *&func){
            if(ordinal==2) {
                (void*&)orig_XInputGetState = func;
                ForceWrite<void*>(func, fake_XInputGetState);
                ret = true;
            }
        }
    );
    return ret;
}
