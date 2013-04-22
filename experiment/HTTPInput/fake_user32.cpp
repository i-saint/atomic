#include <windows.h>
#include "HTTPInput.h"
#include "HTTPInput_Internal.h"

typedef BOOL (WINAPI *GetCursorInfoT)(PCURSORINFO pci);
typedef BOOL (WINAPI *GetKeyboardStateT)(PBYTE lpKeyState);
static GetCursorInfoT orig_GetCursorInfo;
static GetKeyboardStateT orig_GetKeyboardState;

static BOOL WINAPI fake_GetCursorInfo(PCURSORINFO pci)
{
    StartHTTPInputServer();
    orig_GetCursorInfo(pci);
    const HTTPInputData *input = GetHTTPInputData();
    // todo
    return TRUE;
}

static BOOL WINAPI fake_GetKeyboardState(PBYTE lpKeyState)
{
    StartHTTPInputServer();
    orig_GetKeyboardState(lpKeyState);
    const HTTPInputData *input = GetHTTPInputData();
    for(int i=0; i<256; ++i) {
        lpKeyState[i] |= (input->key.keys[i] & 0x80);
    }
    return TRUE;
}

static FuncInfo g_user32_funcs[] = {
    {"GetCursorInfo", 0, (void*)&fake_GetCursorInfo, (void**)&orig_GetCursorInfo},
    {"GetKeyboardState", 0, (void*)&fake_GetKeyboardState, (void**)&orig_GetKeyboardState},
};
OverrideInfo g_user32_overrides = {"user32.dll", _countof(g_user32_funcs), g_user32_funcs};
