#ifndef atm_Types_h
#define atm_Types_h

#include "ist/ist.h"

#define atmSafeRelease(Obj) if(Obj){Obj->release();Obj=NULL;}

#define atmGlobalNamespace(...)  } __VA_ARGS__ namespace atm {
#define atmExportClass(ClassName)        \
    atmGlobalNamespace(                  \
        istSerializeExportClass(atm::ClassName); \
    )

#define atmSerializeRaw(ClassName)       \
    atmGlobalNamespace(                  \
        istSerializeRaw(atm::ClassName); \
    )


namespace atm {

typedef char                int8;
typedef short               int16;
typedef int                 int32;
typedef long long           int64;
typedef unsigned char       uint8;
typedef unsigned short      uint16;
typedef unsigned int        uint32;
typedef unsigned long long  uint64;
typedef __m128i             uint128;
typedef half                float16;
typedef float               float32;
typedef double              float64;

using ist::vec2;
using ist::vec3;
using ist::vec4;
using ist::mat2;
using ist::mat3;
using ist::mat4;
using ist::ivec2;
using ist::ivec3;
using ist::ivec4;
using ist::uvec2;
using ist::uvec3;
using ist::uvec4;
using ist::uvec2;
using ist::uvec3;
using ist::uvec4;
using ist::simdvec4;
using ist::simdmat4;
using ist::soavec24;
using ist::soavec34;
using ist::soavec44;
using ist::atomic_int32;

typedef ist::IBinaryStream Serializer;
typedef ist::IBinaryStream Deserializer;


using ist::SpinMutex;
using ist::Task;
using ist::TaskGroup;
using ist::SFMT;
using ist::FrustumPlanes;
using ist::AABB;
typedef ist::variant16 variant;
typedef ist::variant16 variant16;
typedef ist::variant32 variant32;
typedef ist::variant64 variant64;
typedef ist::variant128 variant128;

namespace i3d = ist::i3dgl;
using namespace ist::i3dgl;

typedef uint32 EntityHandle;
typedef uint32 CollisionHandle;
typedef uint32 CollisionGroup;
enum FunctionID;

typedef uint32 PlayerID;
typedef wchar_t (PlayerName)[16];

enum Interpolation {
    atmE_None,
    atmE_Linear,
    atmE_Decel,
    atmE_Accel,
    atmE_Smooth,
    atmE_Bezier,
};
struct ControlPoint
{
    float32 time;
    float32 value;
    float32 bez_in, bez_out;
    Interpolation interp;

    ControlPoint() : time(0.0f), value(0.0f), bez_in(0.0f),bez_out(0.0f), interp(atmE_Linear) {}
    ControlPoint(float32 t, float32 v, Interpolation ts=atmE_Linear) : time(t), value(v), bez_in(0.0f),bez_out(0.0f), interp(ts) {}
    ControlPoint(float32 t, float32 v, float32 _in, float32 _out, Interpolation ts=atmE_Linear) : time(t), value(v), bez_in(_in),bez_out(_out), interp(ts) {}
    bool operator<(const ControlPoint &p) const { return time<p.time; }
};
atmSerializeRaw(ControlPoint);

class ControlPoints : public ist::raw_vector<ControlPoint>
{
public:
    void addCP(const ControlPoint &v)
    {
        auto i = stl::lower_bound(begin(), end(), v);
        insert(i, v);
    }
    void setCP(uint32 i, const ControlPoint &v)
    {
        if(i<size()) {
            (*this)[i] = v;
            stl::sort(begin(), end());
        }
    }
    void eraseCP(uint32 i)
    {
        if(i<size()) {
            erase(begin()+i);
        }
    }
};


enum ErrorCode {
    atmE_NoError,
    atmE_CreateWindow_Failed,
    atmE_ChangeDisplaySetting_Failed,
    atmE_OpenAL_OpenDevice_Failed,
    atmE_OpenAL_CreateContext_Failed,
    atmE_OpenGL_330NotSupported,
    atmE_OpenGL_Initialization_Failed,
};

class IAtomicGameModule : public boost::noncopyable
{
    istSerializeBlock()
public:
    virtual ~IAtomicGameModule() {}
    virtual void release() { istDelete(this); }

    virtual void initialize() {}
    virtual void finalize() {}

    // フレーム開始時に呼ばれる。
    virtual void frameBegin() {}

    // 同期更新。dt の単位はフレーム。
    virtual void update(float32 dt)=0;

    // 非同期更新。dt の単位はフレーム。
    // asyncupdate() と draw() は並列に走るため、draw() に asyncupdate() の結果に依存する処理がある場合、
    // draw() の中で asyncupdate() の完了を待つ必要がある。
    virtual void asyncupdate(float32 dt)=0;

    // 描画用データを作って Renderer へ送る。(i3d::DeviceContext などを直接触る処理があってはならない)
    // asyncupdate() と draw() は並列に走るため、draw() に asyncupdate() の結果に依存する処理がある場合、
    // draw() の中で asyncupdate() の完了を待つ必要がある。
    virtual void draw() {}

    // フレーム終了時に呼ばれる。
    // 非同期更新処理がある場合、この中で完了を待つこと。(フレームを跨ぐ処理があってはならない)
    virtual void frameEnd() {}
};

} // namespace atm


#endif // atm_Types_h
