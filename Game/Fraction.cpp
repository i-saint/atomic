#include "stdafx.h"
#include "../ist/ist.h"

using namespace ist::graphics;

#include "../types.h"
#include "Message.h"
#include "World.h"
#include "AtomicApplication.h"
#include "Fraction.h"
#include "FractionTask.h"
#include "FractionCollider.h"
#include "../Graphics/Renderer.h"


namespace atomic
{



FractionSet::Interframe::Interframe()
{
    m_update_task = AT_NEW(Task_FractionUpdate) ();
    m_grid = AT_NEW(FractionGrid) ();
}

FractionSet::Interframe::~Interframe()
{
    AT_DELETE(m_grid);
    AT_DELETE(m_update_task);
}

void FractionSet::Interframe::resizeColliders(uint32 block_num)
{
    while(m_collision_results.size() < block_num) {
        QWordVector *rcont = AT_ALIGNED_NEW(QWordVector, 16)();
        m_collision_results.push_back(rcont);

        GridRange grange = {XMVectorSet(0.0f,0.0f,0.0f,0.0f), XMVectorSet(0.0f,0.0f,0.0f,0.0f)};
        m_grid_range.push_back(GridRange());
    }
}


FractionSet::Interframe *FractionSet::s_interframe;



void FractionSet::InitializeInterframe()
{
    if(!s_interframe) {
        s_interframe = AT_ALIGNED_NEW(Interframe, 16)();
    }
}

void FractionSet::FinalizeInterframe()
{
    AT_DELETE(s_interframe);
}




const uint32 FractionSet::BLOCK_SIZE = 512;
const float32 FractionSet::RADIUS = 4.0f;
const float32 FractionSet::BOUNCE = 0.4f;
const float32 FractionSet::MAX_VEL = 1.5f;
const float32 FractionSet::DECELERATE = 0.98f;

FractionSet::FractionSet()
: m_prev(NULL)
, m_idgen(0)
{
}

FractionSet::~FractionSet()
{
    TaskScheduler::waitFor(getInterframe()->getUpdateTask());
}


void FractionSet::initialize( FractionSet* prev, FrameAllocator& alloc )
{
    Task_FractionUpdate *task = getInterframe()->getUpdateTask();
    TaskScheduler::waitFor(task);

    m_prev = prev;
    m_data.clear();

    if(prev) {
        // todo: reserve
        m_data.insert(m_data.begin(), prev->m_data.begin(), prev->m_data.end());
        m_idgen = prev->m_idgen;
    }
    else {
        float32 xv[6] = {500.0f, -500.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float32 yv[6] = {0.0f, 0.0f, 500.0f, -500.0f, 0.0f, 0.0f};
        float32 zv[6] = {0.0f, 0.0f, 0.0f, 0.0f, 500.0f, -500.0f};
        for(uint32 i=0; i<6; ++i) {
            Message_GenerateFraction mes;
            mes.gen_type = Message_GenerateFraction::GEN_SPHERE;
            mes.num = 3500;
            ist::Sphere sphere;
            sphere.x = xv[i];
            sphere.y = yv[i];
            sphere.z = zv[i];
            sphere.r = 200.0f;
            (ist::Sphere&)(*mes.shape_data) = sphere;
            atomicPushMessage(MR_FRACTION, 0, mes);
        }
    }
}


void FractionSet::update()
{
    Task_FractionUpdate *task = getInterframe()->getUpdateTask();
    TaskScheduler::waitFor(task);
    task->initialize(this);
    TaskScheduler::schedule(task);
}

void FractionSet::draw()
{
    Task_FractionUpdate *task = getInterframe()->getUpdateTask();
    TaskScheduler::waitFor(task);

    PassGBuffer_Cube *cube = atomicGetCubeRenderer();
    PassDeferred_SphereLight *light = atomicGetSphereLightRenderer();

    size_t num_data = m_data.size();
    for(uint32 i=0; i<num_data; ++i) {
        cube->pushInstance(m_data[i].pos);
    }
    for(uint32 i=0; i<num_data; i+=200) {
        light->pushInstance(m_data[i].pos);
    }
}


uint32 FractionSet::getNumBlocks() const
{
    const uint32 data_size = m_data.size();
    return data_size/BLOCK_SIZE + (data_size%BLOCK_SIZE!=0 ? 1 : 0);
}

void FractionSet::processGenerateMessage()
{
    MessageIterator<Message_GenerateFraction> mes_gf_iter;
    while(mes_gf_iter.hasNext()) {
        const Message_GenerateFraction& mes = mes_gf_iter.iterate();
        for(uint32 n=0; n<mes.num; ++n) {
            if(mes.gen_type==Message_GenerateFraction::GEN_SPHERE) {
                ist::Sphere& sphere = (ist::Sphere&)(*mes.shape_data);
                FractionData fd;
                fd.id = ++m_idgen;
                fd.alive = 0xFFFFFFFF;
                fd.vel = _mm_set1_ps(0.0f);
                fd.pos = XMVectorSet(sphere.x, sphere.y, sphere.z, 0.0f);

                XMVECTOR r = atomicGenVector3Rand();
                r = XMVectorSubtract(r, _mm_set1_ps(0.5f));
                r = XMVectorMultiply(r, _mm_set1_ps(2.0f));
                r = XMVectorMultiply(r, _mm_set1_ps(sphere.r));
                fd.pos = XMVectorAdd(fd.pos, r);
                fd.vel = XMVectorMultiply(atomicGenVector3Rand(), _mm_set1_ps(1.0f));

                m_data.push_back(fd);
            }
            else if(mes.gen_type==Message_GenerateFraction::GEN_BOX) {
                IST_ASSERT("Message_GenerateFraction::GEN_BOX は未対応");
            }
        }
    }

    const uint32 num_data = m_data.size();
    for(uint32 i=0; i<num_data; ++i) {
        m_data[i].index = i;
    }

    getInterframe()->getGrid()->resizeData(num_data);
    getInterframe()->resizeColliders(getNumBlocks());
}

void FractionSet::move(uint32 block)
{
    // todo: エフェクター処理

    const uint32 num_data = m_data.size();
    const uint32 begin = block*BLOCK_SIZE;
    const uint32 end = std::min<uint32>((block+1)*BLOCK_SIZE, num_data);

    const float32 GRAVITY = 0.07f;
    const float SPHERE_RADIUS = 225.0f;
    XMVECTOR gravity_center     = XMVectorSet(0.0f, 0.0f, 0.0f, 0.0f);
    XMVECTOR gravity_strength   = _mm_set1_ps(GRAVITY);
    XMVECTOR max_speed          = _mm_set1_ps(FractionSet::MAX_VEL);
    XMVECTOR decelerate         = _mm_set1_ps(FractionSet::DECELERATE);
    XMVECTOR sphere_radius      = _mm_set1_ps(SPHERE_RADIUS+RADIUS);
    for(uint32 i=begin; i<end; i+=4) {
        uint32 e = std::min<uint32>(4, end-i);
        FractionData *data = &m_data[i];

        SOAVECTOR3 pos4         = SOAVectorTranspose3(data[0].pos, data[1].pos, data[2].pos, data[3].pos);
        SOAVECTOR3 g_center4    = SOAVectorTranspose3(gravity_center, gravity_center, gravity_center, gravity_center);
        SOAVECTOR3 g_dist       = SOAVectorSubtract3(g_center4, pos4);
        SOAVECTOR3 g_dir        = SOAVectorNormalize3(g_dist);
        SOAVECTOR3 g_accel      = SOAVectorMultiply3S(g_dir, gravity_strength); // 重力加速

        SOAVECTOR3 vel4     = SOAVectorTranspose3(data[0].vel, data[1].vel, data[2].vel, data[3].vel);
        vel4                = SOAVectorAdd3(vel4, g_accel);
        XMVECTOR v_len      = SOAVectorLength3(vel4);
        XMVECTOR v_dec      = XMVectorMultiply(v_len, _mm_set1_ps(FractionSet::DECELERATE));
        XMVECTOR v_select   = XMVectorGreater(v_len, _mm_set1_ps(FractionSet::MAX_VEL));
        XMVECTOR v_speed    = XMVectorSelect(v_len, v_dec, v_select);
        SOAVECTOR3 v_dir    = SOAVectorDivide3S(vel4, v_len);
        SOAVECTOR3 v_next   = SOAVectorMultiply3S(v_dir, v_speed); // 重力加速後の速度

        // 最高速度超えてたら減速

        pos4 = SOAVectorAdd3(pos4, v_next);
        SOAVECTOR3 p_dist       = SOAVectorSubtract3(pos4, g_center4);
        XMVECTOR p_len        = SOAVectorLength3(p_dist);
        XMVECTOR is_bound = XMVectorLess(p_len, sphere_radius); // 跳ね返ったか
        SOAVECTOR3 p_dir        = SOAVectorDivide3S(p_dist, p_len);

        SOAVECTOR4 pos_nextv  = SOAVectorTranspose4(pos4.x, pos4.y, pos4.z);
        for(uint32 i=0; i<e; ++i) {
            data[i].pos = pos_nextv.v[i];
        }

        // 跳ね返ってるのであれば速度を反転
        SOAVECTOR4 v_nextv  = SOAVectorTranspose4(v_next.x, v_next.y, v_next.z);
        SOAVECTOR4 ref_dir = SOAVectorTranspose4(p_dir.x, p_dir.y, p_dir.z);
        uint32 *is_boundv = (uint32*)&is_bound;
        float32 *speedv = (float*)&v_speed;
        for(uint32 i=0; i<e; ++i) {
            if(is_boundv[i]>0) {
                if(speedv[i]<=GRAVITY) {
                    data[i].vel = _mm_set1_ps(0.0f);
                }
                else {
                    XMMATRIX reflection = XMMatrixReflect(ref_dir.v[i]);
                    XMVECTOR tv = XMVector3Transform(v_nextv.v[i], reflection);
                    data[i].vel = XMVectorMultiply(tv, _mm_set1_ps(BOUNCE));
                    //data[i].vel = XMVectorMultiply(ref_dir.v[i], XMVector3Length(v_nextv.v[i]));
                }
                float d = SPHERE_RADIUS + FractionSet::RADIUS;
                data[i].pos = XMVectorMultiply(ref_dir.v[i], _mm_set1_ps(d));
            }
            else {
                data[i].vel = v_nextv.v[i];
            }
        }
    }

    GridRange *grange = getInterframe()->getGridRange(block);
    for(uint32 i=begin; i<end; ++i) {
        grange->range_max = XMVectorMax(grange->range_max, m_data[i].pos);
        grange->range_min = XMVectorMin(grange->range_min, m_data[i].pos);
    }
}

void FractionSet::collisionTest(uint32 block)
{
    const uint32 num_data = m_data.size();
    const uint32 begin = block*BLOCK_SIZE;
    const uint32 end = std::min<size_t>((block+1)*BLOCK_SIZE, num_data);
    QWordVector &results = *getInterframe()->getCollisionResultContainer(block);
    FractionGrid *grid = getInterframe()->getGrid();

    uint32 receiver_num = 0;
    for(uint32 i=begin; i<end; ++i)
    {
        FractionGrid::Data data;
        data.pos = m_data[i].pos;
        data.index = m_data[i].index;

        if(grid->hitTest(results, data)) {
            ++receiver_num;
        }
    }
    {
        FractionGrid::ResultHeader mark_end;
        mark_end.num_collisions = 0;
        results.insert(results.end(), (quadword*)mark_end.v, (quadword*)(mark_end.v + sizeof(mark_end)/16));
    }
}

void FractionSet::collisionProcess(uint32 block)
{
    const uint32 num_data = m_data.size();
    const uint32 begin = block*BLOCK_SIZE;
    const uint32 end = std::min<size_t>((block+1)*BLOCK_SIZE, num_data);
    QWordVector &results = *getInterframe()->getCollisionResultContainer(block);
    if(results.empty()) {
        return;
    }

    const FractionGrid::ResultHeader *rheader = (const FractionGrid::ResultHeader*)&results[0];
    for(;;)
    {
        const FractionGrid::Result *collision = (const FractionGrid::Result*)(rheader+1);
        uint32 num_collisions = rheader->num_collisions;
        if(num_collisions==0) {
            break;
        }

        FractionData &receiver = m_data[rheader->receiver_index];
        XMVECTOR dir = _mm_set1_ps(0.0f);
        XMVECTOR vel = _mm_set1_ps(0.0f);
        for(size_t i=0; i<num_collisions; ++i)
        {
            const FractionGrid::Result& col = *collision;
            dir = XMVectorAdd(dir, col.dir);
            vel = XMVectorAdd(vel, col.vel);
            ++collision;
        }
        const XMVECTOR bounce = _mm_set1_ps(BOUNCE);
        const XMVECTOR neg_bounce = _mm_set1_ps((1.0f-BOUNCE)*0.6f);
        XMMATRIX vref = XMMatrixReflect(XMVector3Normalize(dir));
        XMVECTOR rvel = XMVector3Transform(receiver.vel, vref);
        rvel = XMVectorMultiply(rvel, bounce);
        XMVECTOR add_vel = XMVectorMultiply(vel, neg_bounce);
        XMVECTOR next_vel = XMVectorAdd(rvel, add_vel);
        receiver.vel = next_vel;
        receiver.pos = XMVectorAdd(receiver.pos, XMVectorMultiply(dir, XMVector3Length(next_vel)));

        rheader = (const FractionGrid::ResultHeader*)collision;
    }

    results.clear();
}

void FractionSet::updateGrid()
{
    GridRange range = {XMVectorSet(0.0f,0.0f,0.0f,0.0f), XMVectorSet(0.0f,0.0f,0.0f,0.0f)};
    uint32 num_blocks = getNumBlocks();
    for(uint32 i=0; i<num_blocks; ++i) {
        GridRange *grange = getInterframe()->getGridRange(i);
        range.range_max = XMVectorMax(range.range_max, grange->range_max);
        range.range_min = XMVectorMin(range.range_min, grange->range_min);
    }
    // todo: グリッドサイズが 0 になるの禁止

    FractionGrid *grid = getInterframe()->getGrid();
    grid->setGridRange(range.range_min, range.range_max);
    grid->clear();
    size_t num_data = m_data.size();
    for(size_t i=0; i<num_data; ++i) {
        grid->setData(i, m_data[i].pos, m_data[i].vel);
    }
}


size_t FractionSet::getRequiredMemoryOnNextFrame()
{
    // todo:
    return 0;
}

}
