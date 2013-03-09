#include "stdafx.h"
#include "types.h"
#include "Game/AtomicApplication.h"
#include "FunctionID.h"
#include "ist/Base/BinaryCall.h"

istImplementOperatorNewDelete();

namespace atomic {
    void InitializeCrashReporter();
    void FinalizeCrashReporter();
} // namespace atomic
using namespace atomic;

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
int Func1(int a) { return a*2; }
int Func2(int a, int b) { return a*b; }

struct Test
{
    int value;
    ivec4 iv;
    void MemFn0() { value=2; }
    int MemFn2(int a, int b) { return a*b*2; }
    int ConstMemFn0() const { return 3; }
    int ConstMemFn2(int a, int b) const { return a*b*4; }
    const ivec4& TestRef(const ivec4 &a, const ivec4 &b) { iv=a*b; return iv; }
};

void test()
{
    //atomic::FunctionIDEachPair([](const ist::EnumStr &es){
    //    istPrint("%d:%s\n", es.num, es.str);
    //});

    Test obj;
    {
        int ret;
        ist::ArgList<int,int> args(2, 4);
        ist::BinaryCall(Func0, &ret);
        assert(ret==1);

        ist::BinaryCall(Func2, &ret, &args);
        assert(ret==8);

        ist::BinaryCall(&Test::MemFn0, obj, NULL, NULL);
        assert(obj.value==2);

        ist::BinaryCall(&Test::MemFn2, obj, &ret, &args);
        assert(ret==16);

        ist::BinaryCall(&Test::ConstMemFn0, obj, &ret, NULL);
        assert(ret==3);

        ist::BinaryCall(&Test::ConstMemFn2, obj, &ret, &args);
        assert(ret==32);
    }
    {
        ivec4 fv1(1, 2, 3, 4);
        ivec4 fv2(5, 6, 7, 8);
        ist::ArgList<const ivec4&, const ivec4&> args(fv1, fv2);
        ist::ArgHolder<const ivec4&> ret;

        ist::BinaryCallRef(&Test::TestRef, obj, &ret, &args);
        const ivec4 &r = ret;
        assert(r[0]==5);
        assert(r[1]==12);
        assert(r[2]==21);
        assert(r[3]==32);
    }
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

