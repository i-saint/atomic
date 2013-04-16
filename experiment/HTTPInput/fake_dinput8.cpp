#include <windows.h>
#include <dinput.h>
#include "HTTPInput.h"

typedef HRESULT (WINAPI *DirectInput8CreateT)(HINSTANCE hinst, DWORD dwVersion, REFIID riidltf, LPVOID * ppvOut, LPUNKNOWN punkOuter);
typedef HRESULT (__stdcall *CreateDeviceT)(IDirectInput8*, REFGUID, LPDIRECTINPUTDEVICE8A*, LPUNKNOWN);
typedef HRESULT (__stdcall *GetDeviceStateT)(IDirectInputDevice8*, DWORD, LPVOID);

DirectInput8CreateT orig_DirectInput8Create;
CreateDeviceT orig_CreateDevice;
GetDeviceStateT orig_GetDeviceState;

HRESULT __stdcall fake_GetDeviceState(IDirectInputDevice8 *dev, DWORD size, LPVOID data)
{
    HRESULT r = orig_GetDeviceState(dev, size, data);
    if(SUCCEEDED(r)) {
        if(size==sizeof(DIJOYSTATE)) {
            DIJOYSTATE &state = *(DIJOYSTATE*)data;
            // todo:
        }
        else if(size==sizeof(DIJOYSTATE2)) {
            DIJOYSTATE2 &state = *(DIJOYSTATE2*)data;
            // todo:
        }
    }
    return r;
}

HRESULT __stdcall fake_CreateDevice(IDirectInput8 *di8, REFGUID guid, LPDIRECTINPUTDEVICE8A *out, LPUNKNOWN unk)
{
    HRESULT r = orig_CreateDevice(di8, guid, out, unk);
    if(SUCCEEDED(r)) {
        void **&vftable = ((void***)(*out))[0];
        if(orig_GetDeviceState==NULL) { (void*&)orig_GetDeviceState=vftable[9]; }
        vftable[9] = &fake_GetDeviceState;
    }
    return r;
}

HRESULT WINAPI fake_DirectInput8Create(HINSTANCE hinst, DWORD dwVersion, REFIID riidltf, LPVOID * ppvOut, LPUNKNOWN punkOuter)
{
    HRESULT r = orig_DirectInput8Create(hinst, dwVersion, riidltf, ppvOut, punkOuter);
    if(SUCCEEDED(r)) {
        void **&vftable = ((void***)(*ppvOut))[0];
        if(orig_CreateDevice==NULL) { (void*&)orig_CreateDevice=vftable[3]; }
        vftable[3] = &fake_CreateDevice;
        StartHTTPInputServer();
    }
    return r;
}

bool HookDirectInput8()
{
    bool ret = false;
    EachImportFunctionInEveryModule("dinput8.dll", [&](const char *funcname, void *&func){
        if(strcmp(funcname, "DirectInput8Create")==0) {
            (void*&)orig_DirectInput8Create = func;
            ForceWrite<void*>(func, fake_DirectInput8Create);
            ret = true;
        }
    });
    return ret;
}
