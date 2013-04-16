#include <windows.h>
#include <mmsystem.h>
#include "HTTPInput.h"

typedef MMRESULT (WINAPI *joyGetPosExT)(UINT uJoyID, LPJOYINFOEX pji);
joyGetPosExT orig_joyGetPosEx;

MMRESULT WINAPI fake_joyGetPosEx(UINT uJoyID, LPJOYINFOEX pji)
{
    StartHTTPInputServer();
    orig_joyGetPosEx(uJoyID, pji);
    const HTTPInputData *input = GetHTTPInputData();
    // todo
    return JOYERR_NOERROR;
}

bool HookMMJoustick()
{
    bool ret = false;
    EachImportFunctionInEveryModule("winmm.dll", [&](const char *funcname, void *&func){
        if(strcmp(funcname, "joyGetPosEx")==0) {
            (void*&)orig_joyGetPosEx = func;
            ForceWrite<void*>(func, fake_joyGetPosEx);
            ret = true;
        }
    });
    return ret;
}
