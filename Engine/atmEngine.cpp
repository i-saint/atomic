#include "atmPCH.h"
#include "types.h"
#include "Engine/Game/AtomicApplication.h"
#include "FunctionID.h"
#include "Util.h"
#include "ist/Base/BinaryCall.h"
#include "Poco/DirectoryIterator.h"

istImplementOperatorNewDelete();


namespace atm {
void atmInitializeCrashReporter();
void atmFinalizeCrashReporter();
} // namespace atm

void atmExecApplication(int argc, char* argv[])
{
    atm::AtomicApplication app;
    if(app.initialize(argc, argv)) {
        app.mainLoop();
    }
    app.finalize();
}

using namespace ist;

atmCLinkage atmAPI int32 atmMain(int argc, char* argv[])
{
    dpInitialize(dpConfig(dpE_LogSimple, dpE_SysDefault, "atomic.dpconf"));
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

int istmain(int argc, char* argv[])
{
    return atmMain(argc, argv);
}
