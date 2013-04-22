#include <windows.h>
#include <xinput.h>
#include "HTTPInput.h"
#include "HTTPInput_Internal.h"

typedef DWORD (WINAPI *XInputGetStateT)(DWORD dwUserIndex, XINPUT_STATE* pState);
typedef DWORD (WINAPI *XInputGetKeystrokeT)(DWORD dwUserIndex, DWORD dwReserved, PXINPUT_KEYSTROKE pKeystroke);
static XInputGetStateT orig_XInputGetState;
static XInputGetKeystrokeT orig_XInputGetKeystroke;

static DWORD WINAPI fake_XInputGetState(DWORD dwUserIndex, XINPUT_STATE* pState)
{
    DWORD r = orig_XInputGetState(dwUserIndex, pState);
    if(HTTPInput_GetConfig()->override_xinput) {
        HTTPInput_StartServer();
        if(r!=ERROR_SUCCESS) {
            memset(pState, 0, sizeof(XINPUT_STATE));
        }
        if(dwUserIndex<HTTPInputData::MaxPads) {
            XINPUT_GAMEPAD &pad = pState->Gamepad;
            const HTTPInputData::Pad &vpad = HTTPInput_GetData()->pad[dwUserIndex];
            pad.sThumbLX = abs(vpad.x1+INT16_MIN)>abs((int)pad.sThumbLX) ?  vpad.x1+INT16_MIN : pad.sThumbLX;
            pad.sThumbLY = abs(vpad.y1+INT16_MIN)>abs((int)pad.sThumbLY) ?-(vpad.y1-INT16_MAX): pad.sThumbLY;
            for(int32 i=0; i<4; ++i) {
                pad.wButtons |= (vpad.buttons[i] & 0x80 ? 1 : 0)<<12;
            }
        }
        return ERROR_SUCCESS;
    }
    return r;
}

static DWORD WINAPI fake_XInputGetKeystroke(DWORD dwUserIndex, DWORD dwReserved, PXINPUT_KEYSTROKE pKeystroke)
{
    DWORD ret = orig_XInputGetKeystroke(dwUserIndex, dwReserved, pKeystroke);
    if(HTTPInput_GetConfig()->override_xinput) {
        HTTPInput_StartServer();
        // todo
    }
    return ret;
}

static FuncInfo g_xinput_funcs[] = {
    {"XInputGetState", 2, (void*)&fake_XInputGetState, (void**)&orig_XInputGetState},
    {"XInputGetKeystroke", 0, (void*)&fake_XInputGetKeystroke, (void**)&orig_XInputGetKeystroke},
};
OverrideInfo g_xinput_overrides = {"xinput1_\\d+\\.dll", _countof(g_xinput_funcs), g_xinput_funcs};
