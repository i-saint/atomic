#include "stdafx.h"
#include "types.h"
#include "Game/AtomicApplication.h"
#include "FunctionID.h"
#include "ist/Base/BinaryCall.h"

istImplementOperatorNewDelete();

#ifdef _M_X64
#   define dpPlatform "x64"
#else
#   define dpPlatform "Win32"
#endif
#ifdef _DEBUG
#   define dpConfiguration "Debug"
#else
#   define dpConfiguration "Release"
#endif
#define dpObjDir "_tmp/" dpConfiguration "/atomic" 


namespace atm {

void atmInitializeCrashReporter();
void atmFinalizeCrashReporter();

void atmExecApplication(int argc, char* argv[])
{
    atm::AtomicApplication app;
    if(app.initialize(argc, argv)) {
        app.mainLoop();
    }
    app.finalize();
}

atmAPI int32 atmMain(int argc, char* argv[])
{
    dpInitialize();
    ist::forceLink();

    atm::atmInitializeCrashReporter();
    istCrashReportBegin
    atmExecApplication(argc, argv);
    istCrashReportRescue
    istCrashReportEnd
    atm::atmFinalizeCrashReporter();

    dpFinalize();
    return 0;
}

} // namespace atm
using namespace atm;

int istmain(int argc, char* argv[])
{
    return atmMain(argc, argv);
}
