#include "stdafx.h"
#include "../ist/ist.h"
#include "../types.h"
#include "Message.h"
#include "World.h"
#include "Fraction.h"
#include "FractionTask.h"
#include "FractionCollider.h"

namespace atomic {



FractionGrid::FractionGrid()
{
    setGridRange(XMVectorSet(-500.0f, -500.0f, -500.0f, 0.0f), XMVectorSet( 500.0f,  500.0f,  500.0f, 0.0f));
}

XMVECTORI32 FractionGrid::getCoord( XMVECTOR pos ) const
{
    XMVECTOR r1 = XMVectorSubtract(pos, m_range_min);
    XMVECTOR r2 = XMVectorAdd(r1, m_range_max);
    XMVECTOR r3 = XMVectorDivide(r2, m_grid_size);
    XMVECTOR r4 = XMVectorMax(r3, XMVectorSet(0.0f, 0.0f, 0.0f, 0.0f));
    XMVECTOR r5 = XMVectorMin(r4, XMVectorSet((float32)(GRID_NUM_X-1), (float32)(GRID_NUM_Y-1), (float32)(GRID_NUM_X-1), 0.0f));
    float32 *f = (float32*)&r5;

    XMVECTORI32 index = {int32(f[0]), int32(f[1]), int32(f[2]), 0};
    return index;
}

void FractionGrid::setGridRange( XMVECTOR rmin, XMVECTOR rmax )
{
    m_range_min = rmin;
    m_range_max = rmax;
    m_grid_size = XMVectorDivide(XMVectorSubtract(rmax, rmin), XMVectorSet((float32)GRID_NUM_X, (float32)GRID_NUM_Y, (float32)GRID_NUM_Z, 0.0f));
}

void FractionGrid::clear()
{
    for(uint32 yi=0; yi<GRID_NUM_Y; ++yi) {
        for(uint32 zi=0; zi<GRID_NUM_Z; ++zi) {
            for(uint32 xi=0; xi<GRID_NUM_X; ++xi) {
                m_blocks[yi][zi][xi].clear();
            }
        }
    }
}

void FractionGrid::pushData( const Data &d )
{
    XMVECTORI32 i = getCoord(d.v);
    m_blocks[i.i[1]][i.i[2]][i.i[0]].push(d);
}

uint32 FractionGrid::hitTest( QWordVector &out, const Data &receiver ) const
{
    __declspec(thread) static stl::vector<Result> *s_tmp_result = NULL;
    if(s_tmp_result==NULL) {
        s_tmp_result = AT_NEW(stl::vector<Result>) stl::vector<Result>();
        s_tmp_result->reserve(128);
    }

    FractionSet *fraction_set = GetFractions();
    const XMVECTOR diameter = _mm_set_ps1(FractionSet::RADIUS*2.0f);
    const XMVECTOR receiver_pos1 = receiver.v;
    const SOAVECTOR3 receiver_pos = SOAVectorSet3(
        _mm_set_ps1(XMVectorGetX(receiver_pos1)),
        _mm_set_ps1(XMVectorGetY(receiver_pos1)),
        _mm_set_ps1(XMVectorGetZ(receiver_pos1))
        );

    const XMVECTORI32 range_min = getCoord(XMVectorSubtract(receiver_pos1, diameter));
    const XMVECTORI32 range_max = getCoord(XMVectorAdd(receiver_pos1, diameter));
    for(int32 yi=range_min.i[1]; yi<=range_max.i[1]; ++yi) {
        for(int32 zi=range_min.i[2]; zi<=range_max.i[2]; ++zi) {
            for(int32 xi=range_min.i[0]; xi<=range_max.i[0]; ++xi) {

                const Block *block = &m_blocks[yi][zi][xi];
                const SOAVECTOR4 *data = block->data;
                uint32 num_senders = block->num_data;

                for(uint32 si=0; si<num_senders; si+=4) {
                    const SOAVECTOR4 tpos= SOAVectorTranspose4(data->x, data->y, data->z, data->w);
                    const XMVECTOR   indices =tpos.w;
                    const SOAVECTOR3 dist= SOAVectorSubtract3(receiver_pos, tpos);
                    const XMVECTOR   len = SOAVectorLength3(dist);
                    const SOAVECTOR3 dir = SOAVectorDivide3S(dist, len);
                    const XMVECTOR   hit = XMVectorLessOrEqual(len, diameter);

                    const SOAVECTOR4 dirv = SOAVectorTranspose4(dir.x, dir.y, dir.z);
                    const uint32* hitv = (const uint32*)&hit;
                    const uint32* indicesv = (const uint32*)&indices;
                    uint32 e = std::min<uint32>(4, num_senders-si);
                    for(size_t i=0; i<e; ++i) {
                        if(hitv[i] && receiver.index!=indicesv[i]) {
                            Result r;
                            r.dir = dirv.v[i];
                            r.receiver_index = receiver.index;
                            r.sender_index = indicesv[i];
                            r.vel = fraction_set->getFraction(indicesv[i])->vel;
                            s_tmp_result->push_back(r);
                        }
                    }
                    ++data;
                }
            }
        }
    }

    uint32 n = s_tmp_result->size();
    if(n > 0) {
        ResultHeader rh;
        rh.receiver_index = receiver.index;
        rh.num_chunks = 1;
        rh.num_collisions = n;
        out.insert(out.end(), (quadword*)rh.v, (quadword*)(rh.v + sizeof(ResultHeader)/16));
        out.insert(out.end(), (quadword*)(*s_tmp_result)[0].v, (quadword*)((*s_tmp_result)[0].v + n*sizeof(Result)/16));
    }
    s_tmp_result->clear();

    return n;
}




void FractionCollider::beginPushData(const FractionData& receiver)
{
    m_tmp_data_header.pos = receiver.pos;
    m_tmp_data_header.receiver_index = receiver.index;
}

void FractionCollider::pushData(const FractionData& target)
{
    Data data;
    data.pos = target.pos;
    data.vel = target.vel;
    data.sender_index = target.index;
    m_tmp_data.push_back(data);
}

void FractionCollider::endPushData()
{
    if(!m_tmp_data.empty()) {
        uint32 n = m_tmp_data.size();
        m_tmp_data_header.num_target = n;
        m_data.insert(m_data.end(), (quadword*)m_tmp_data_header.v, (quadword*)(m_tmp_data_header.v + sizeof(DataHeader)/16));
        m_data.insert(m_data.end(), (quadword*)m_tmp_data[0].v, (quadword*)(m_tmp_data[0].v + (sizeof(Data)/16)*n));
        ++m_num_data_chunk;
    }
    m_tmp_data_header.receiver_index = 0xffffffff;
    m_tmp_data.clear();
}


void FractionCollider::process()
{
    m_num_result_chunk = 0;
    m_result.clear();

    uint32 total_result = 0;
    const uint32 num_chunks = m_num_data_chunk;
    const DataHeader* dheader = getData();
    for(uint32 ichunk=0; ichunk<num_chunks; ++ichunk) {
        m_tmp_result_header.receiver_index = dheader->receiver_index;
        m_tmp_result_header.num_collision = 0;


        const XMVECTOR pos1 = dheader->pos;
        const XMVECTOR union_radius = _mm_set_ps1(FractionSet::RADIUS*2.0f);
        const SOAVECTOR3 pos = SOAVectorSet3(
            _mm_set_ps1(XMVectorGetX(pos1)),
            _mm_set_ps1(XMVectorGetY(pos1)),
            _mm_set_ps1(XMVectorGetZ(pos1))
            );

        const size_t num_target = dheader->num_target;
        const Data* data= (const Data*)(dheader+1);
        for(uint32 idata=0; idata<num_target; idata+=4) {
            const SOAVECTOR3 tpos= SOAVectorTranspose3(data[0].pos, data[1].pos, data[2].pos, data[3].pos);
            const SOAVECTOR3 dist= SOAVectorSubtract3(pos, tpos);
            const XMVECTOR   len = SOAVectorLength3(dist);
            const SOAVECTOR3 dir = SOAVectorDivide3S(dist, len);
            const XMVECTOR   hit = XMVectorLessOrEqual(len, union_radius);

            const SOAVECTOR4 dirv = SOAVectorTranspose4(dir.x, dir.y, dir.z);
            const uint32* hitv = (const uint32*)&hit;
            uint32 e = std::min<uint32>(4, num_target-idata);
            for(size_t i=0; i<e; ++i) {
                if(hitv[i]) {
                    {
                        Result r;
                        r.dir = dirv.v[i];
                        r.vel = data[i].vel;
                        r.receiver_index = dheader->receiver_index;
                        m_tmp_result.push_back(r);
                    }
                }
            }
            data += e;
        }
        dheader = (const DataHeader*)data;

        if(!m_tmp_result.empty()) {
            uint32 n = m_tmp_result.size();
            total_result += n;
            m_tmp_result_header.num_collision = n;
            m_result.insert(m_result.end(), (quadword*)m_tmp_result_header.v, (quadword*)(m_tmp_result_header.v + sizeof(ResultHeader)/16));
            m_result.insert(m_result.end(), (quadword*)m_tmp_result[0].v, (quadword*)(m_tmp_result[0].v + n*sizeof(Result)/16));
            ++m_num_result_chunk;
        }
        m_tmp_result.clear();
    }

    m_num_data_chunk = 0;
    m_data.clear();
}




} // namespace atomic
