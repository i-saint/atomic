#ifndef __atomic_FractionCollider__
#define __atomic_FractionCollider__

namespace atomic {

struct FractionData;

class __declspec(align(16)) FractionGrid : public boost::noncopyable
{
public:
    static const uint32 GRID_NUM_X = 75;
    static const uint32 GRID_NUM_Y = 75;
    static const uint32 GRID_NUM_Z = 75;

    union Data
    {
        struct {
            float32 pos_x;
            float32 pos_y;
            float32 pos_z;
            uint32 index;
        };
        XMVECTOR v;
    };

    class __declspec(align(16)) Block
    {
    public:
        static const uint32 INITAL_CAPACITY = 16;

        SOAVECTOR4 *data;
        uint32 capacity;
        uint32 num_data;

        Block()
            : data(NULL)
            , capacity(INITAL_CAPACITY)
            , num_data(0)
        {
            data = (SOAVECTOR4*)AT_ALIGNED_MALLOC(sizeof(SOAVECTOR4)*capacity/4, 16);
        }

        ~Block()
        {
            AT_FREE(data);
        }

        void push(const Data& d)
        {
            if(num_data==capacity) {
                SOAVECTOR4 *old = data;
                capacity = capacity*2;
                data = (SOAVECTOR4*)AT_ALIGNED_MALLOC(sizeof(SOAVECTOR4)*capacity/4, 16);
                memcpy(data, old, sizeof(SOAVECTOR4)*capacity/4/2);
                AT_FREE(old);
            }
            uint32 i1 = num_data/4;
            uint32 i2 = num_data%4;
            data[i1].v[i2] = d.v;
            num_data++;
        }

        void clear()
        {
            num_data = 0;
        }
    };

    // 衝突結果データ
    // [ResultHeader][Result][Result]... というメモリ配置にします
    union __declspec(align(16)) ResultHeader
    {
        struct {
            id_t receiver_index;
            uint32 num_collisions;
            uint32 num_chunks;
        };
        XMVECTOR v[1];
    };
    union __declspec(align(16)) Result
    {
        struct {
            union {
                XMVECTOR dir;
                struct {
                    float32 dirv[3];
                    uint32 receiver_index;
                };
            };
            union {
                XMVECTOR vel;
                struct {
                    float32 velv[3];
                    uint32 sender_index;
                };
            };
        };
        XMVECTOR v[2];
    };

private:
    Block m_blocks[GRID_NUM_Y][GRID_NUM_Z][GRID_NUM_X];
    XMVECTOR m_range_min;
    XMVECTOR m_range_max;
    XMVECTOR m_grid_size;

    uint32 m_num_chunks[GRID_NUM_Y];

public:
    FractionGrid();

    uint32 getNumBlocks() const { return GRID_NUM_Y*GRID_NUM_Z*GRID_NUM_X; }
    void setGridRange(XMVECTOR rmin, XMVECTOR rmax);

    void clear();
    void pushData(const Data &d);

    XMVECTORI32 getCoord(XMVECTOR pos) const;
    // ret: num collision
    uint32 hitTest(QWordVector &out, const Data &data) const;
};



class FractionCollider : public boost::noncopyable
{
public:
    // 衝突元オブジェクトデータ
    // [DataHeader][Data][Data]... というメモリ配置にします
    union __declspec(align(16)) DataHeader
    {
        struct {
            union {
                struct {
                    id_t receiver_index;
                    uint32 num_target;
                };
                XMVECTOR pad;
            };
            XMVECTOR pos;
        };
        XMVECTOR v[2];
    };
    union __declspec(align(16)) Data
    {
        struct {
            XMVECTOR pos;
            union {
                XMVECTOR vel;
                struct {
                    float32 velv[3];
                    uint32 sender_index;
                };
            };
        };
        XMVECTOR v[2];
    };

    // 衝突結果データ
    // [ResultHeader][Result][Result]... というメモリ配置にします
    union __declspec(align(16)) ResultHeader
    {
        struct {
            id_t receiver_index;
            size_t num_collision;
        };
        XMVECTOR v[1];
    };
    union __declspec(align(16)) Result
    {
        struct {
            union {
                XMVECTOR dir;
                struct {
                    float32 dirv[3];
                    uint32 receiver_index;
                };
            };
            union {
                XMVECTOR vel;
                struct {
                    float32 velv[3];
                    uint32 sender_index;
                };
            };
        };
        XMVECTOR v[2];
    };

private:
    size_t                  m_num_data_chunk;
    size_t                  m_num_result_chunk;
    stl::vector<quadword> m_data;
    stl::vector<quadword> m_result;

    DataHeader              m_tmp_data_header;
    ResultHeader            m_tmp_result_header;
    stl::vector<Data>     m_tmp_data;
    stl::vector<Result>   m_tmp_result;


public:
    FractionCollider()
        : m_num_data_chunk(0)
        , m_num_result_chunk(0)
    {
        m_data.reserve(1024*128);   // 2MB
        m_result.reserve(1024*128);
    }

    size_t getDataChunkNum() const          { return m_num_data_chunk; }
    const DataHeader* getData() const       { return (const DataHeader*)&m_data[0]; }
    size_t getResultChunkNum() const        { return m_num_result_chunk; }
    const ResultHeader* getResult() const   { return (const ResultHeader*)&m_result[0]; }


    void beginPushData(const FractionData& receiver);
    void pushData(const FractionData& target);
    void endPushData();
    void process();
};

} // namespace atomic
#endif // __atomic_FractionCollider__
