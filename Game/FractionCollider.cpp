#include "stdafx.h"
#include "ist/ist.h"
#include "types.h"
#include "Game/Message.h"
#include "Game/Fraction.h"
#include "Game/FractionTask.h"
#include "Game/FractionCollider.h"

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


void FractionGrid::resizeData(uint32 n)
{
    //m_data.resize(n);
}

void FractionGrid::pushData(uint32 id, XMVECTOR pos, XMVECTOR vel)
{
    Data d;
    d.pos = pos;
    d.vel = vel;
    d.id = id;
    XMVECTORI32 it = getCoord(pos);
    m_blocks[it.i[1]][it.i[2]][it.i[0]].push(d);
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

uint32 FractionGrid::hitTest( QWordVector &out, FractionData &receiver ) const
{
    __declspec(thread) static stl::vector<Result> *s_tmp_result = NULL;
    if(s_tmp_result==NULL) {
        s_tmp_result = IST_NEW(stl::vector<Result>) ();
        s_tmp_result->reserve(128);
    }

    receiver.density = 0.0f;

    const float32 radius2f = FractionSet::RADIUS*2.0f;
    const float32 rcp_radius2f = 1.0f/(radius2f);
    const float32 h_sqf = FractionSet::SMOOTH_LENGTH*FractionSet::SMOOTH_LENGTH;
    const XMVECTOR zero = _mm_set_ps1(0.0f);
    const XMVECTOR h_sq = _mm_set_ps1(h_sqf);
    const XMVECTOR mass = _mm_set_ps1(FractionSet::MASS);
    const XMVECTOR radius2 = _mm_set_ps1(radius2f);
    const XMVECTOR radius2_sq = _mm_set_ps1(radius2f*radius2f);
    const XMVECTOR rcp_radius2 = _mm_set_ps1(rcp_radius2f);
    const XMVECTOR receiver_pos1 = receiver.pos;
    const SOAVECTOR3 receiver_pos = SOAVectorSet3(
        _mm_set_ps1(XMVectorGetX(receiver_pos1)),
        _mm_set_ps1(XMVectorGetY(receiver_pos1)),
        _mm_set_ps1(XMVectorGetZ(receiver_pos1))
        );

    // 衝突候補オブジェクト群のグリッドのインデクス
    const XMVECTORI32 range_min = getCoord(XMVectorSubtract(receiver_pos1, radius2));
    const XMVECTORI32 range_max = getCoord(XMVectorAdd(receiver_pos1, radius2));

    // 以下、 SoA に並び替えつつ衝突判定。
    // 衝突していたら 相手→自分への方向を算出。
    for(int32 yi=range_min.i[1]; yi<=range_max.i[1]; ++yi) {
        for(int32 zi=range_min.i[2]; zi<=range_max.i[2]; ++zi) {
            for(int32 xi=range_min.i[0]; xi<=range_max.i[0]; ++xi) {

                const Block *block = &m_blocks[yi][zi][xi];
                const Data *data = block->data;
                uint32 num_senders = block->num_data;
                const XMVECTOR rid = XMVectorSetInt(receiver.id, receiver.id, receiver.id, receiver.id);

                for(uint32 si=0; si<num_senders; si+=4) {
                    // 位置の w 要素は id
                    const SOAVECTOR4 tpos= SOAVectorTranspose4(data[0].pos, data[1].pos, data[2].pos, data[3].pos);
                    const SOAVECTOR3 dist= SOAVectorSubtract3(receiver_pos, tpos);
                    const XMVECTOR   len_sq = SOAVectorLengthSquare3(dist);
                    // 衝突の方向は 半径*2 の逆数を利用して割り算を使わず低精度＆高速に算出
                    const SOAVECTOR3 dir = SOAVectorMultiply3S(dist, rcp_radius2);
                    // めりこんでいる ＆ id が一緒 (=自分自身との衝突) ではない場合衝突
                    const XMVECTOR   hit = XMVectorAndInt(XMVectorLessOrEqual(len_sq, radius2_sq), XMVectorNotEqualInt(rid, tpos.w));

                    const SOAVECTOR4 dirv = SOAVectorTranspose4(dir.x, dir.y, dir.z);

                    // 濃度算出
                    const XMVECTOR r_sq = XMVectorMax(XMVectorSubtract(h_sq, len_sq), zero);
                    const XMVECTOR density = XMVectorMultiply( XMVectorMultiply(XMVectorMultiply(r_sq, r_sq), r_sq),  mass);

                    // _mm_movemask_ps() 使った場合↓逆に遅くなった…
                    //const uint32 hitv = _mm_movemask_ps(hit);
                    //const uint32 eq_ridv = _mm_movemask_ps(eq_rid);
                    //uint32 e = std::min<uint32>(4, num_senders-si);
                    //for(size_t i=0; i<e; ++i) {
                    //    if((hitv&1<<i)!=0 && (eq_ridv&1<<i)==0) {

                    const uint32* hitv = (const uint32*)&hit;
                    const float32* densityv = (const float*)&density;
                    uint32 e = std::min<uint32>(4, num_senders-si);
                    for(size_t i=0; i<e; ++i) {
                        receiver.density += densityv[i];
                        if(hitv[i]) {
                            Result r;
                            r.dir = dirv.v[i];
                            r.vel = data[i].vel;
                            s_tmp_result->push_back(r);
                        }
                    }
                    data+=4;
                }
            }
        }
    }

    uint32 n = s_tmp_result->size();
    if(n > 0) {
        ResultHeader rh;
        rh.receiver_index = receiver.index;
        rh.num_collisions = n;
        out.insert(out.end(), (quadword*)rh.v, (quadword*)(rh.v + sizeof(rh)/16));
        out.insert(out.end(), (quadword*)(*s_tmp_result)[0].v, (quadword*)((*s_tmp_result)[0].v + n*sizeof(Result)/16));
    }
    s_tmp_result->clear();


    // 圧力など算出
    {
        // Precompute kernel coefficients
        static const float32 h                 = FractionSet::SMOOTH_LENGTH;
        static const float32 poly6_coef        = 315.0f/(64.0f*XM_PI*pow(h, 9));
        static const float32 grad_poly6_coef   = 945.0f/(32.0f*XM_PI*pow(h, 9));
        static const float32 lap_poly6_coef    = 945.0f/(32.0f*XM_PI*pow(h, 9));
        static const float32 grad_spiky_coef   = -45.0f/(XM_PI*pow(h, 6));
        static const float32 lap_vis_coef      = 45.0f/(XM_PI*pow(h, 6));
    }

    return n;
}



} // namespace atomic
