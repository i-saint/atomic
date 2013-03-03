#include "stdafx.h"
#include "types.h"
#include "Game/AtomicApplication.h"
#include "FunctionID.h"
#include "ist/Base/VariantCall.h"

namespace atomic {
    void InitializeCrashReporter();
    void FinalizeCrashReporter();
} // namespace atomic

void ExecApp(int argc, char* argv[])
{
    // クラッシュさせるテスト
    //*static_cast<int*>(NULL) = 0;

    atomic::AtomicApplication app;
    if(app.initialize(argc, argv)) {
        app.mainLoop();
    }
    app.finalize();
}

int Func0() { return 1; }
int Func2(int a, int b) { return a*b; }

struct Test
{
    int value;
    void MemFn0() { value=2; }
    int MemFn2(int a, int b) { return a*b*2; }
    int ConstMemFn0() const { return 3; }
    int ConstMemFn2(int a, int b) const { return a*b*4; }
};

void test()
{
    using namespace atomic;
    //atomic::FunctionIDEachPair([](const ist::EnumStr &es){
    //    istPrint("%d:%s\n", es.num, es.str);
    //});

    Test obj;
    variant ret;
    variant args[2] = {int32(2), int(4)};
    ist::VariantCall(Func0, ret);
    assert(ret.cast<int32>()==1);

    ist::VariantCall(Func2, ret, args);
    assert(ret.cast<int32>()==8);

    ist::VariantCall(&Test::MemFn0, obj, ret);
    assert(obj.value==2);

    ist::VariantCall(&Test::MemFn2, obj, ret, args);
    assert(ret.cast<int32>()==16);

    ist::VariantCall(&Test::ConstMemFn0, obj, ret);
    assert(ret.cast<int32>()==3);

    ist::VariantCall(&Test::ConstMemFn2, obj, ret, args);
    assert(ret.cast<int32>()==32);
}

int istmain(int argc, char* argv[])
{
    //test();

    atomic::InitializeCrashReporter();
istCrashReportBegin
    ExecApp(argc, argv);
istCrashReportRescue
istCrashReportEnd
    atomic::FinalizeCrashReporter();
    return 0;
}

