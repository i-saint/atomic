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

    union __declspec(align(16)) Data
    {
        struct {
            float32 pos_x;
            float32 pos_y;
            float32 pos_z;
            uint32 id;
            float32 vel_x;
            float32 vel_y;
            float32 vel_z;
            float32 vel_w;
        };
        struct {
            XMVECTOR pos;
            XMVECTOR vel;
        };
        XMVECTOR v[2];
    };

    class __declspec(align(16)) Block
    {
    public:
        static const uint32 INITAL_CAPACITY = 16;

        Data *data;
        uint32 capacity;
        uint32 num_data;

        Block()
            : data(NULL)
            , capacity(INITAL_CAPACITY)
            , num_data(0)
        {
            data = (Data*)AT_ALIGNED_MALLOC(sizeof(Data)*capacity, 16);
        }

        ~Block()
        {
            AT_FREE(data);
        }

        void push(const Data &d)
        {
            if(num_data==capacity) {
                Data *old = data;
                capacity = capacity*2;
                data = (Data*)AT_ALIGNED_MALLOC(sizeof(Data)*capacity, 16);
                memcpy(data, old, sizeof(Data)*capacity/2);
                AT_FREE(old);
            }
            data[num_data++] = d;
        }

        void clear()
        {
            num_data = 0;
        }

        const Data* getData() const { return data; }
    };

    union __declspec(align(16)) ResultHeader
    {
        struct {
            id_t receiver_index;
            uint32 num_collisions;
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
            };
        };
        XMVECTOR v[2];
    };

private:
    Block m_blocks[GRID_NUM_Y][GRID_NUM_Z][GRID_NUM_X];
    XMVECTOR m_range_min;
    XMVECTOR m_range_max;
    XMVECTOR m_grid_size;

    // アクセス頻度低いデータは直接グリッドには突っ込まずこちらへ
    //stl::vector<Data> m_data;

public:
    FractionGrid();

    uint32 getNumBlocks() const { return GRID_NUM_Y*GRID_NUM_Z*GRID_NUM_X; }
    void setGridRange(XMVECTOR rmin, XMVECTOR rmax);

    void resizeData(uint32 n);
    void pushData(uint32 id, XMVECTOR pos, XMVECTOR vel);
    void clear();

    XMVECTORI32 getCoord(XMVECTOR pos) const;
    // ret: num collision
    uint32 hitTest(QWordVector &out, const FractionData &data) const;
};




} // namespace atomic
#endif // __atomic_FractionCollider__
