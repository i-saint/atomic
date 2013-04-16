#include <windows.h>
#include "HTTPInput.h"

typedef BOOL (WINAPI *GetCursorInfoT)(PCURSORINFO pci);
typedef BOOL (WINAPI *GetKeyboardStateT)(PBYTE lpKeyState);
GetCursorInfoT orig_GetCursorInfo;
GetKeyboardStateT orig_GetKeyboardState;

BOOL WINAPI fake_GetCursorInfo(PCURSORINFO pci)
{
    StartHTTPInputServer();
    orig_GetCursorInfo(pci);
    const HTTPInputData *input = GetHTTPInputData();
    // todo
    return TRUE;
}

BOOL WINAPI fake_GetKeyboardState(PBYTE lpKeyState)
{
    StartHTTPInputServer();
    orig_GetKeyboardState(lpKeyState);
    const HTTPInputData *input = GetHTTPInputData();
    // todo
    return TRUE;
}

bool HookKeyboardMouse()
{
    bool ret = false;
    EachImportFunctionInEveryModule("user32.dll", [&](const char *funcname, void *&func){
        if(strcmp(funcname, "GetCursorInfo")==0) {
            (void*&)orig_GetCursorInfo = func;
            ForceWrite<void*>(func, fake_GetCursorInfo);
            ret = true;
        }
        else if(strcmp(funcname, "GetKeyboardState")==0) {
            (void*&)orig_GetKeyboardState = func;
            ForceWrite<void*>(func, fake_GetKeyboardState);
            ret = true;
        }
    });
    return ret;
}
