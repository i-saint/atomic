#ifndef atomic_Types_h
#define atomic_Types_h

#include "ist/ist.h"

namespace atomic {

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
enum FunctionID;

typedef uint32 PlayerID;
typedef wchar_t (PlayerName)[16];


enum ATOMIC_ERROR {
    ATERR_NOERROR,
    ATERR_CREATEWINDOW_FAILED,
    ATERR_CHANGEDISPLAYSETTINGS_FAILED,
    ATERR_OPENAL_OPENDEVICE_FAILED,
    ATERR_OPENAL_CREATECONTEXT_FAILED,
    ATERR_OPENGL_330_IS_NOT_SUPPORTED,
    ATERR_OPENGL_INITIALIZATION_FAILED,
    ATERR_CUDA_NO_DEVICE,
    ATERR_CUDA_INSUFFICIENT_DRIVER,
};

class IAtomicGameModule : public boost::noncopyable
{
public:
    virtual ~IAtomicGameModule() {}

    // フレーム開始時に呼ばれる。
    virtual void frameBegin()=0;

    // 同期更新。dt の単位はフレーム。
    virtual void update(float32 dt)=0;

    // 非同期更新。dt の単位はフレーム。
    // asyncupdate() と draw() は並列に走るため、draw() に asyncupdate() の結果に依存する処理がある場合、
    // draw() の中で asyncupdate() の完了を待つ必要がある。
    virtual void asyncupdate(float32 dt)=0;

    // 描画用データを作って Renderer へ送る。(i3d::DeviceContext などを直接触る処理があってはならない)
    // asyncupdate() と draw() は並列に走るため、draw() に asyncupdate() の結果に依存する処理がある場合、
    // draw() の中で asyncupdate() の完了を待つ必要がある。
    virtual void draw()=0;

    // フレーム終了時に呼ばれる。
    // 非同期更新処理がある場合、この中で完了を待つこと。(フレームを跨ぐ処理があってはならない)
    virtual void frameEnd()=0;

    istSerializeBlock(
        )
};


struct EntitiesQueryContext
{
    ist::raw_vector<uint32> id;
    ist::raw_vector<uint32> type;
    ist::raw_vector<vec2>   size;
    ist::raw_vector<vec2>   pos;

    void clear()
    {
        id.clear();
        type.clear();
        size.clear();
        pos.clear();
    }

    size_t sizeByte() const
    {
        return
            id.size()*sizeof(uint32) +
            type.size()*sizeof(uint32) +
            size.size()*sizeof(vec2) +
            pos.size()*sizeof(vec2);
    }
};

} // namespace atomic

#define atomicSafeRelease(Obj) if(Obj){Obj->release();Obj=NULL;}

#define atomicGlobalNamespace(...)  } __VA_ARGS__ namespace atomic {
#define atomicExportClass(ClassName)        \
    atomicGlobalNamespace(                  \
        istSerializeExportClass(ClassName); \
    )

#define atomicSerializeRaw(ClassName)   \
atomicGlobalNamespace(                  \
    istSerializeRaw(ClassName);         \
)


#endif // atomic_Types_h
