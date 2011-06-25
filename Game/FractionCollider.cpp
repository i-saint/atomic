#include "stdafx.h"
#include "../ist/ist.h"
#include "../types.h"
#include "Message.h"
#include "World.h"
#include "Fraction.h"
#include "FractionTask.h"
#include "FractionCollider.h"

namespace atomic
{

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
