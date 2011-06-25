#ifndef __atomic_FractionCollider__
#define __atomic_FractionCollider__


namespace atomic
{


class FractionCollider
{
public:
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
    eastl::vector<quadword> m_data;
    eastl::vector<quadword> m_result;

    DataHeader              m_tmp_data_header;
    ResultHeader            m_tmp_result_header;
    eastl::vector<Data>     m_tmp_data;
    eastl::vector<Result>   m_tmp_result;


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
