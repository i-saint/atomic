#include "istPCH.h"
#include "ist/stdex/crtex.h"
#include "Process.h"

namespace ist {

#if defined(ist_env_Windows) && !defined(ist_env_Master)

bool ExecVCTool( const char *params )
{
    static stl::string vcvars;
    if(vcvars.empty()) {
        stl::string VCVersion;
#if     _MSC_VER==1500
        VCVersion = "9.0";
#elif   _MSC_VER==1600
        VCVersion = "10.0";
#elif   _MSC_VER==1700
        VCVersion = "11.0";
#elif   _MSC_VER==1800
        VCVersion = "12.0";
#else
#   error
#endif
        stl::string keyName = "SOFTWARE\\Microsoft\\VisualStudio\\SxS\\VC7";
        char value[MAX_PATH];
        DWORD size = MAX_PATH;
        HKEY key;
        LONG retKey = ::RegOpenKeyExA(HKEY_LOCAL_MACHINE, keyName.c_str(), 0, KEY_READ|KEY_WOW64_32KEY, &key);
        LONG retVal = ::RegQueryValueExA(key, VCVersion.c_str(), NULL, NULL, (LPBYTE)value, &size );
        if( retKey==ERROR_SUCCESS && retVal==ERROR_SUCCESS  ) {
            vcvars += '"';
            vcvars += value;
            vcvars += "vcvarsall.bat";
            vcvars += '"';
#ifdef _WIN64
            vcvars += " amd64";
#else // _WIN64
            vcvars += " x86";
#endif // _WIN64
            vcvars += " && ";
        }
        ::RegCloseKey( key );
    }

    stl::string command = vcvars + params;
    STARTUPINFOA si; 
    PROCESS_INFORMATION pi; 
    istMemset(&si, 0, sizeof(si)); 
    istMemset(&pi, 0, sizeof(pi)); 
    si.cb = sizeof(si);
    if(::CreateProcessA(NULL, (LPSTR)command.c_str(), NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi)==TRUE) {
        DWORD exit_code = 0;
        ::WaitForSingleObject(pi.hThread, INFINITE);
        ::WaitForSingleObject(pi.hProcess, INFINITE);
        ::GetExitCodeProcess(pi.hProcess, &exit_code);
        ::CloseHandle(pi.hThread);
        ::CloseHandle(pi.hProcess);
        ::Sleep(100); // 終了直後だとファイルの書き込みが終わってないことがあるっぽい？ので少し待つ…
        return exit_code==0;
    }
    return false;
}
#endif // defined(ist_env_Windows) && !defined(ist_env_Master)

} // namespace ist

