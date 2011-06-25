#include "stdafx.h"
#include "../ist/ist.h"

using namespace ist::graphics;

#include "../types.h"
#include "Message.h"
#include "World.h"
#include "Fraction.h"
#include "FractionTask.h"
#include "FractionCollider.h"


namespace atomic
{

template<class T>
struct LessX
{
    bool operator()(const T& a, const T& b) const
    {
        return a.x < b.x;
    }
};

template<class T>
struct LessY
{
    bool operator()(const T& a, const T& b) const
    {
        return a.y < b.y;
    }
};

template<class T>
struct LessZ
{
    bool operator()(const T& a, const T& b) const
    {
        return a.z < b.z;
    }
};




void SetFloat3(float (&v)[3], float x, float y, float z)
{
    v[0] = x;
    v[1] = y;
    v[2] = z;
}
void SetFloat3(float (&v)[3], float (&s)[3])
{
    v[0] = s[0];
    v[1] = s[1];
    v[2] = s[2];
}

void CreateCubeModel(ModelData& model)
{
    float vertex[24][3];
    float normal[24][3];
    int index[24];

    float n[3];
    float ur[3];
    float bl[3];
    SetFloat3(ur,  3, 3, 3);
    SetFloat3(bl, -3,-3,-3);

    SetFloat3(n, 1.0f, 0.0f, 0.0f);
    SetFloat3(normal[0], n);
    SetFloat3(normal[1], n);
    SetFloat3(normal[2], n);
    SetFloat3(normal[3], n);
    SetFloat3(vertex[0], ur[0], ur[1], ur[2]);
    SetFloat3(vertex[1], ur[0], bl[1], ur[2]);
    SetFloat3(vertex[2], ur[0], bl[1], bl[2]);
    SetFloat3(vertex[3], ur[0], ur[1], bl[2]);

    SetFloat3(n, -1.0f, 0.0f, 0.0f);
    SetFloat3(normal[4], n);
    SetFloat3(normal[5], n);
    SetFloat3(normal[6], n);
    SetFloat3(normal[7], n);
    SetFloat3(vertex[4], bl[0], ur[1], ur[2]);
    SetFloat3(vertex[5], bl[0], ur[1], bl[2]);
    SetFloat3(vertex[6], bl[0], bl[1], bl[2]);
    SetFloat3(vertex[7], bl[0], bl[1], ur[2]);

    SetFloat3(n, 0.0f, 1.0f, 0.0f);
    SetFloat3(normal[8], n);
    SetFloat3(normal[9], n);
    SetFloat3(normal[10], n);
    SetFloat3(normal[11], n);
    SetFloat3(vertex[8], ur[0], ur[1], ur[2]);
    SetFloat3(vertex[9], ur[0], ur[1], bl[2]);
    SetFloat3(vertex[10], bl[0], ur[1], bl[2]);
    SetFloat3(vertex[11], bl[0], ur[1], ur[2]);

    SetFloat3(n, 0.0f, -1.0f, 0.0f);
    SetFloat3(normal[12], n);
    SetFloat3(normal[13], n);
    SetFloat3(normal[14], n);
    SetFloat3(normal[15], n);
    SetFloat3(vertex[12], ur[0], bl[1], ur[2]);
    SetFloat3(vertex[13], bl[0], bl[1], ur[2]);
    SetFloat3(vertex[14], bl[0], bl[1], bl[2]);
    SetFloat3(vertex[15], ur[0], bl[1], bl[2]);

    SetFloat3(n, 0.0f, 0.0f, 1.0f);
    SetFloat3(normal[16], n);
    SetFloat3(normal[17], n);
    SetFloat3(normal[18], n);
    SetFloat3(normal[19], n);
    SetFloat3(vertex[16], ur[0], ur[1], ur[2]);
    SetFloat3(vertex[17], bl[0], ur[1], ur[2]);
    SetFloat3(vertex[18], bl[0], bl[1], ur[2]);
    SetFloat3(vertex[19], ur[0], bl[1], ur[2]);

    SetFloat3(n, 0.0f, 0.0f, -1.0f);
    SetFloat3(normal[20], n);
    SetFloat3(normal[21], n);
    SetFloat3(normal[22], n);
    SetFloat3(normal[23], n);
    SetFloat3(vertex[20], ur[0], ur[1], bl[2]);
    SetFloat3(vertex[21], ur[0], bl[1], bl[2]);
    SetFloat3(vertex[22], bl[0], bl[1], bl[2]);
    SetFloat3(vertex[23], bl[0], ur[1], bl[2]);

    for(size_t i=0; i<24; ++i) {
        index[i] = i;
    }

    model.setVertex(vertex, 24, ModelData::VTX_FLOAT3, ModelData::USAGE_STATIC);
    model.setNormal(normal, 24, ModelData::USAGE_STATIC);
    model.setIndex(index, 24, ModelData::IDX_INT32, ModelData::PRM_QUADS, ModelData::USAGE_STATIC);
}



FractionSet::Interframe::Interframe()
{
    CreateCubeModel(m_model);
    m_update_task = EA_NEW(Task_FractionUpdate) Task_FractionUpdate();
}

FractionSet::Interframe::~Interframe()
{
    EA_DELETE(m_update_task);

    for(uint32 i=0; i<m_colliders.size(); ++i) {
        EA_DELETE(m_colliders[0]);
    }
    m_colliders.clear();
    CreateCubeModel(m_model);
}

void FractionSet::Interframe::resizeColliders(uint32 block_num)
{
    while(m_colliders.size() < block_num) {
        FractionCollider *idata = EA_ALIGNED_NEW(FractionCollider, 16) FractionCollider();
        m_colliders.push_back(idata);
    }
}

FractionCollider* FractionSet::Interframe::getCollider(uint32 block)
{
    return m_colliders[block];
}


FractionSet::Interframe *FractionSet::s_interframe;



void FractionSet::InitializeInterframe()
{
    if(!s_interframe) {
        s_interframe = EA_ALIGNED_NEW(Interframe, 16) Interframe();
    }
}

void FractionSet::FinalizeInterframe()
{
    EA_DELETE(s_interframe);
}




const uint32 FractionSet::BLOCK_SIZE = 512;
const float32 FractionSet::RADIUS = 4.0f;
const float32 FractionSet::BOUNCE = 0.4f;
const float32 FractionSet::MAX_VEL = 1.5f;
const float32 FractionSet::DECELERATE = 0.995f;

FractionSet::FractionSet(FractionSet* prev, FrameScopedAllocator* alloc)
: m_prev(prev), m_next(NULL)
, m_alloc(alloc)
, m_idgen(0)
, m_num_dead(0)
{
    float32 xv[6] = {500.0f, -500.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float32 yv[6] = {0.0f, 0.0f, 500.0f, -500.0f, 0.0f, 0.0f};
    float32 zv[6] = {0.0f, 0.0f, 0.0f, 0.0f, 500.0f, -500.0f};
    for(uint32 i=0; i<6; ++i) {
        Message_GenerateFraction mes;
        mes.type = MES_GENERATE_FRACTION;
        mes.gen_type = Message_GenerateFraction::GEN_SPHERE;
        mes.num = 3500;
        ist::Sphere sphere;
        sphere.x = xv[i];
        sphere.y = yv[i];
        sphere.z = zv[i];
        sphere.r = 200.0f;
        (ist::Sphere&)(*mes.shape_data) = sphere;
        pushGenerateMessage(mes);

    }
}

FractionSet::~FractionSet()
{
    TaskScheduler *scheduler = TaskScheduler::getInstance();
    scheduler->waitFor(getInterframe()->getUpdateTask());

    if(m_prev) { m_prev->m_next = m_next; }
    if(m_next) { m_next->m_prev = m_prev; }
}



void FractionSet::update()
{
    TaskScheduler *scheduler = TaskScheduler::getInstance();
    Task_FractionUpdate *task = getInterframe()->getUpdateTask();
    scheduler->waitFor(task);
    task->initialize(this);
    scheduler->schedule(task);
}

void FractionSet::sync()
{
    TaskScheduler *scheduler = TaskScheduler::getInstance();
    Task_FractionUpdate *task = getInterframe()->getUpdateTask();
    scheduler->waitFor(task);
}

void FractionSet::flushMessage()
{
    // todo: メッセージ全送信
    // todo: 破壊メッセージ処理
    // todo: 次フレームを作成して生成メッセージを受け渡す
}

void FractionSet::processMessage()
{
}

void FractionSet::draw()
{
    ModelData *model = getInterframe()->getCubeModel();
    // todo: 頂点データとか作成して描画コマンド発行
    for(uint32 i=0; i<m_data.size(); ++i) {
        FractionData& data = m_data[i];
        if(data.alive) {
            const float *pos = (float*)&data.pos;
            glPushMatrix();
            glTranslatef(pos[0], pos[1], pos[2]);
            model->draw();
            glPopMatrix();
        }
    }
}


uint32 FractionSet::getNumBlocks() const
{
    const uint32 data_size = m_data.size();
    return data_size/BLOCK_SIZE + (data_size%BLOCK_SIZE!=0 ? 1 : 0);
}

void FractionSet::processGenerateMessage()
{
    for(uint32 i=0; i<m_gen_mes.size(); ++i) {
        Message_GenerateFraction& mes = m_gen_mes[i];
        for(uint32 n=0; n<mes.num; ++n) {
            if(mes.gen_type==Message_GenerateFraction::GEN_SPHERE) {
                ist::Sphere& sphere = (ist::Sphere&)(*mes.shape_data);
                FractionData fd;
                fd.id = ++m_idgen;
                fd.alive = 0xFFFFFFFF;
                fd.vel = _mm_set1_ps(0.0f);
                fd.pos = XMVectorSet(sphere.x, sphere.y, sphere.z, 0.0f);

                XMVECTOR r = GenVector3Rand();
                r = XMVectorSubtract(r, _mm_set1_ps(0.5f));
                r = XMVectorMultiply(r, _mm_set1_ps(2.0f));
                r = XMVectorMultiply(r, _mm_set1_ps(sphere.r));
                fd.pos = XMVectorAdd(fd.pos, r);
                fd.vel = XMVectorMultiply(GenVector3Rand(), _mm_set1_ps(1.0f));

                m_data.push_back(fd);
            }
            else if(mes.gen_type==Message_GenerateFraction::GEN_BOX) {
                IST_ASSERT("Message_GenerateFraction::GEN_BOX は未対応");
            }
        }
    }
    m_gen_mes.clear();

    uint32 dead = 0;
    m_num_dead = 0;
    const uint32 num_data = m_data.size();
    for(uint32 i=0; i<num_data; ++i) {
        m_data[i].index = i;
        if(m_data[i].alive==0) {
            ++dead;
        }
    }
}

void FractionSet::move(uint32 block)
{
    // todo: エフェクター処理

    const uint32 num_data = m_data.size();
    const uint32 begin = block*BLOCK_SIZE;
    const uint32 end = std::min<uint32>((block+1)*BLOCK_SIZE, num_data);

    const float SPHERE_RADIUS = 225.0f;
    XMVECTOR gravity_center     = XMVectorSet(0.0f, 0.0f, 0.0f, 0.0f);
    XMVECTOR gravity_strength   = _mm_set1_ps(0.05f);
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
        uint32* is_boundv = (uint32*)&is_bound;
        SOAVECTOR4 ref_dir = SOAVectorTranspose4(p_dir.x, p_dir.y, p_dir.z);
        for(uint32 i=0; i<e; ++i) {
            if(is_boundv[i]>0) {
                XMMATRIX reflection = XMMatrixReflect(ref_dir.v[i]);
                XMVECTOR tv = XMVector3Transform(v_nextv.v[i], reflection);
                data[i].vel = XMVectorMultiply(tv, _mm_set1_ps(BOUNCE));
                //data[i].vel = XMVectorMultiply(ref_dir.v[i], XMVector3Length(v_nextv.v[i]));
                float d = SPHERE_RADIUS + FractionSet::RADIUS;
                data[i].pos = XMVectorMultiply(ref_dir.v[i], _mm_set1_ps(d));
            }
            else {
                data[i].vel = v_nextv.v[i];
            }
        }
    }
}


void FractionSet::collisionTest(size_t block)
{
    const uint32 num_data = m_data.size();
    const uint32 begin = block*BLOCK_SIZE;
    const uint32 end = std::min<size_t>((block+1)*BLOCK_SIZE, num_data);

    FractionCollider *collider = getInterframe()->getCollider(block);

    for(uint32 i=begin; i<end; ++i)
    {
        FractionData& receiver = m_data[i];
        uint32 xbegin, xend;
        uint32 ybegin, yend;
        uint32 zbegin, zend;

            uint32 xindex   = receiver.xindex;
            XHolder xlh     = m_xorder[xindex];
            XHolder xhh     = m_xorder[xindex];
            XHolder *xob    = m_xorder.begin();
            XHolder *xoe    = m_xorder.end();
            XHolder *xl, *xh;
            xlh.x  -= RADIUS * 2.0f;
            xhh.x  += RADIUS * 2.0f;
            xl      = eastl::lower_bound(xob, xob+xindex, xlh, LessX<XHolder>());
            xh      = eastl::lower_bound(xob+xindex, xoe, xhh, LessX<XHolder>());
            xbegin  = xl==xob+xindex ? xindex : m_data[xl->index].xindex;
            xend    = xh==xoe ? num_data : m_data[xh->index].xindex;

            uint32 yindex   = receiver.yindex;
            YHolder ylh     = m_yorder[yindex];
            YHolder yhh     = m_yorder[yindex];
            YHolder *yob    = m_yorder.begin();
            YHolder *yoe    = m_yorder.end();
            YHolder *yl, *yh;
            ylh.y  -= RADIUS * 2.0f;
            yhh.y  += RADIUS * 2.0f;
            yl      = eastl::lower_bound(yob, yob+yindex, ylh, LessY<YHolder>());
            yh      = eastl::lower_bound(yob+yindex, yoe, yhh, LessY<YHolder>());
            ybegin  = yl==yob+yindex ? yindex : m_data[yl->index].yindex;
            yend    = yh==yoe ? num_data : m_data[yh->index].yindex;

            uint32 zindex   = receiver.zindex;
            ZHolder zlh     = m_zorder[zindex];
            ZHolder zhh     = m_zorder[zindex];
            ZHolder *zob    = m_zorder.begin();
            ZHolder *zoe    = m_zorder.end();
            ZHolder *zl, *zh;
            zlh.z  -= RADIUS * 2.0f;
            zhh.z  += RADIUS * 2.0f;
            zl      = eastl::lower_bound(zob, zob+zindex, zlh, LessZ<ZHolder>());
            zh      = eastl::lower_bound(zob+zindex, zoe, zhh, LessZ<ZHolder>());
            zbegin  = zl==zob+zindex ? zindex : m_data[zl->index].zindex;
            zend    = zh==zoe ? num_data : m_data[zh->index].zindex;

        collider->beginPushData(receiver);
        id_t self_id   = receiver.id;
        for(uint32 xi=xbegin; xi<xend; ++xi) {
            FractionData& target = m_data[m_xorder[xi].index];
            if(  target.id!=self_id &&
                (target.yindex>=ybegin && target.yindex<yend) &&
                (target.zindex>=zbegin && target.zindex<zend) )
            {
                collider->pushData(target);
            }
        }
        collider->endPushData();
    }

    collider->process();
}

void FractionSet::collisionProcess(size_t block)
{
    FractionCollider *collider = getInterframe()->getCollider(block);
    uint32 receiver_num = collider->getResultChunkNum();

    const FractionCollider::ResultHeader *rheader = collider->getResult();
    for(size_t header_i=0; header_i<receiver_num; ++header_i)
    {
        const FractionCollider::Result *collision = (const FractionCollider::Result*)(rheader+1);
        FractionData &receiver = m_data[rheader->receiver_index];
        uint32 num_collision = rheader->num_collision;
        XMVECTOR dir = _mm_set1_ps(0.0f);
        XMVECTOR vel = _mm_set1_ps(0.0f);
        for(size_t i=0; i<num_collision; ++i)
        {
            const FractionCollider::Result& col = *collision;
            dir = XMVectorAdd(dir, col.dir);
            vel = XMVectorAdd(vel, col.vel);

            ++collision;
        }
        //dir = XMVector3Normalize(dir);
        const XMVECTOR bounce = _mm_set1_ps(BOUNCE);
        const XMVECTOR neg_bounce = _mm_set1_ps((1.0f-BOUNCE)*0.6f);
        XMMATRIX vref = XMMatrixReflect(XMVector3Normalize(dir));
        XMVECTOR rvel = XMVector3Transform(receiver.vel, vref);
        rvel = XMVectorMultiply(rvel, bounce);
        XMVECTOR add_vel = XMVectorMultiply(vel, neg_bounce);
        XMVECTOR next_vel = XMVectorAdd(rvel, add_vel);
        receiver.vel = next_vel;
        receiver.pos = XMVectorAdd(receiver.pos, XMVectorMultiply(dir, XMVector3Length(next_vel)));

        rheader = (const FractionCollider::ResultHeader*)collision;
    }
}


void FractionSet::sortXOrder()
{
    size_t num_data = m_data.size();
    m_xorder.resize(num_data);
    for(size_t i=0; i<num_data; ++i) {
        XHolder xo;
        xo.index= i;
        xo.x    = XMVectorGetX(m_data[i].pos);
        m_xorder[i] = xo;
    }

    eastl::stable_sort(m_xorder.begin(), m_xorder.end(), LessX<XHolder>());
    for(size_t i=0; i<num_data; ++i) {
        m_data[m_xorder[i].index].xindex = i;
    }
}

void FractionSet::sortYOrder()
{
    size_t num_data = m_data.size();
    m_yorder.resize(num_data);
    for(size_t i=0; i<num_data; ++i) {
        YHolder yo;
        yo.index= i;
        yo.y    = XMVectorGetY(m_data[i].pos);
        m_yorder[i] = yo;
    }

    eastl::stable_sort(m_yorder.begin(), m_yorder.end(), LessY<YHolder>());
    for(size_t i=0; i<num_data; ++i) {
        m_data[m_yorder[i].index].yindex = i;
    }
}

void FractionSet::sortZOrder()
{
    size_t num_data = m_data.size();
    m_zorder.resize(num_data);
    for(size_t i=0; i<num_data; ++i) {
        ZHolder zo;
        zo.index= i;
        zo.z    = XMVectorGetZ(m_data[i].pos);
        m_zorder[i] = zo;
    }

    eastl::stable_sort(m_zorder.begin(), m_zorder.end(), LessZ<ZHolder>());
    for(size_t i=0; i<num_data; ++i) {
        m_data[m_zorder[i].index].zindex = i;
    }
}


size_t FractionSet::getRequiredMemoryOnNextFrame()
{
    // todo:
    return 0;
}

}
