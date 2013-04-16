#include <windows.h>
#include <xinput.h>
#include "HTTPInput.h"

typedef DWORD (WINAPI *XInputGetStateT)(DWORD dwUserIndex, XINPUT_STATE* pState);
XInputGetStateT orig_XInputGetState;

DWORD WINAPI fake_XInputGetState(DWORD dwUserIndex, XINPUT_STATE* pState)
{
    StartHTTPInputServer();
    orig_XInputGetState(dwUserIndex, pState);
    const HTTPInputData *input = GetHTTPInputData();
    // todo
    return ERROR_SUCCESS;
}

bool HookXInput()
{
    bool ret = false;
    EachImportFunctionInEveryModule("xinput.dll", [&](const char *funcname, void *&func){
        if(strcmp(funcname, "XInputGetState")==0) {
            (void*&)orig_XInputGetState = func;
            ForceWrite<void*>(func, fake_XInputGetState);
            ret = true;
        }
    });
    return ret;
}
