#include "stdafx.h"
#include "ist/ist.h"
#include "types.h"
#include "AtomicApplication.h"
#include "Graphics/Renderer.h"
#include "Game/Message.h"
#include "Game/World.h"
#include "Game/Fraction.h"
#include "Game/FractionTask.h"
#include "Game/FractionCollider.h"

namespace atomic
{



FractionSet::Interframe::Interframe()
{
    m_task_beforedraw = IST_NEW16(Task_FractionBeforeDraw)();
    m_task_afterdraw = IST_NEW16(Task_FractionAfterDraw)();
    m_task_copy = IST_NEW16(Task_FractionCopy)();
    m_grid = IST_NEW16(FractionGrid)();

    SPHInitialize();
}

FractionSet::Interframe::~Interframe()
{
    IST_DELETE(m_grid);
    IST_DELETE(m_task_copy);
    IST_DELETE(m_task_afterdraw);
    IST_DELETE(m_task_beforedraw);
}

void FractionSet::Interframe::resizeColliders(uint32 block_num)
{
}


FractionSet::Interframe *FractionSet::s_interframe;

void FractionSet::InitializeInterframe()
{
    if(!s_interframe) {
        s_interframe = IST_NEW16(Interframe)();
    }
}

void FractionSet::FinalizeInterframe()
{
    IST_DELETE(s_interframe);
}




const uint32 FractionSet::BLOCK_SIZE = 512;
const float32 FractionSet::RADIUS = 0.04f;
const float32 FractionSet::BOUNCE = 0.5f;
const float32 FractionSet::MAX_VEL = 10.5f;
const float32 FractionSet::DECELERATE = 0.98f;
const float32 FractionSet::GRIDSIZE = 5.12f;
const uint32 FractionSet::NUM_GRID_CELL = 128;

const XMVECTOR g_grid_pos = _mm_set_ps1(FractionSet::GRIDSIZE/2.0f);
const XMVECTOR g_rcp_cell_length = _mm_set_ps1(1.0f / (FractionSet::GRIDSIZE/FractionSet::NUM_GRID_CELL));
const XMVECTOR g_grid_zero = _mm_set_ps1(0.0f);
const XMVECTOR g_grid_max = _mm_set_ps1(FractionSet::NUM_GRID_CELL-1);

const float32 g_fSmoothlen = 0.04f;
const float32 g_fPressureStiffness = 200.0f;
const float32 g_fRestDensity = 1000.0f;
const float32 g_fParticleMass = 0.0002f;
const float32 g_fViscosity = 0.1f;
const float32 g_fDensityCoef = g_fParticleMass * 315.0f / (64.0f * XM_PI * pow(g_fSmoothlen, 9));
const float32 g_fGradPressureCoef = g_fParticleMass * -45.0f / (XM_PI * pow(g_fSmoothlen, 6));
const float32 g_fLapViscosityCoef = g_fParticleMass * g_fViscosity * 45.0f / (XM_PI * pow(g_fSmoothlen, 6));

//
//inline uint32 GenerateHashFromPosition(XMVECTOR pos)
//{
//    XMVECTOR grid = XMVectorClamp(XMVectorMultiply(XMVectorAdd(pos, g_grid_pos), g_rcp_cell_length), g_grid_zero, g_grid_max);
//    uint32 ux = _mm_cvtss_si32(grid);
//    uint32 uy = _mm_cvtss_si32(_mm_shuffle_ps(grid, grid, SSE_SHUFFLE(1,1,1,1)));
//    uint32 uz = _mm_cvtss_si32(_mm_shuffle_ps(grid, grid, SSE_SHUFFLE(2,2,2,2)));
//    const uint32 cells = FractionSet::NUM_GRID_CELL;
//    const uint32 hash = uz*(cells*cells) + uy*cells + ux;
//    return hash;
//}
//
//inline ivec3 GetIndexFromPositionHash(uint32 phash)
//{
//    const uint32 mask = FractionSet::NUM_GRID_CELL-1;
//    return ivec3(
//        (phash >> 0) & mask,
//        (phash >> 7) & mask,
//        (phash >>14) & mask );
//}
//
//inline uint32 GetHashFromIndex(uint32 x, uint32 y, uint32 z)
//{
//    const uint32 cells = FractionSet::NUM_GRID_CELL;
//    return z*(cells*cells) + y*cells + x;
//}
//
//inline float CalculateDensity(float r_sq)
//{
//    const float h_sq = g_fSmoothlen * g_fSmoothlen;
//    // Implements this equation:
//    // W_poly6(r, h) = 315 / (64 * pi * h^9) * (h^2 - r^2)^3
//    // g_fDensityCoef = fParticleMass * 315.0f / (64.0f * PI * fSmoothlen^9)
//    return g_fDensityCoef * (h_sq - r_sq) * (h_sq - r_sq) * (h_sq - r_sq);
//}
//
//inline float CalculatePressure(float density)
//{
//    // Implements this equation:
//    // Pressure = B * ((rho / rho_0)^y  - 1)
//    return g_fPressureStiffness * stl::max<float32>(pow(density / g_fRestDensity, 3) - 1, 0);
//}
//
//inline XMVECTOR CalculateGradPressure(float r, float P_pressure, float N_pressure, float N_density, XMVECTOR diff)
//{
//    const float h = g_fSmoothlen;
//    float avg_pressure = 0.5f * (N_pressure + P_pressure);
//    // Implements this equation:
//    // W_spkiey(r, h) = 15 / (pi * h^6) * (h - r)^3
//    // GRAD( W_spikey(r, h) ) = -45 / (pi * h^6) * (h - r)^2
//    // g_fGradPressureCoef = fParticleMass * -45.0f / (PI * fSmoothlen^6)
//    return g_fGradPressureCoef * avg_pressure / N_density * (h - r) * (h - r) / r * (diff);
//}
//
//inline XMVECTOR CalculateLapVelocity(float r, XMVECTOR P_velocity, XMVECTOR N_velocity, float N_density)
//{
//    const float h = g_fSmoothlen;
//    XMVECTOR l = _mm_set_ps1(g_fLapViscosityCoef / N_density * (h - r));
//    XMVECTOR vel_diff = XMVectorSubtract(N_velocity, P_velocity);
//    // Implements this equation:
//    // W_viscosity(r, h) = 15 / (2 * pi * h^3) * (-r^3 / (2 * h^3) + r^2 / h^2 + h / (2 * r) - 1)
//    // LAPLACIAN( W_viscosity(r, h) ) = 45 / (pi * h^6) * (h - r)
//    // g_fLapViscosityCoef = fParticleMass * fViscosity * 45.0f / (PI * fSmoothlen^6)
//    return  XMVectorMultiply(l, vel_diff);
//}



FractionSet::FractionSet()
: m_prev(NULL)
, m_next(NULL)
, m_idgen(0)
{
    m_grid.resize(128*128*128);
}

FractionSet::~FractionSet()
{
    sync();
}


void FractionSet::initialize()
{
    m_prev = NULL;

    static bool s_init = false;
    if(!s_init) {
        s_init = true;
        float32 xv[4] = {2.5f, -2.5f, 0.0f, 0.0f};
        float32 yv[4] = {0.0f, 0.0f, 2.5f, -2.5f};
        for(uint32 i=0; i<4; ++i) {
            Message_GenerateFraction mes;
            mes.gen_type = Message_GenerateFraction::GEN_SPHERE;
            mes.num = 3000;

            ist::Sphere sphere;
            sphere.x = xv[i];
            sphere.y = yv[i];
            sphere.z = 0.0f;
            sphere.r = 1.2f;
            mes.assignData<ist::Sphere>(sphere);
            atomicPushMessage(MR_FRACTION, 0, mes);
        }
    }
}


void FractionSet::update()
{
    Task_FractionBeforeDraw *task = getInterframe()->getTask_BeforeDraw();
    task->waitForComplete();
    task->initialize(this);
    task->kick();
}

void FractionSet::sync() const
{
    Task_FractionBeforeDraw *task_before= getInterframe()->getTask_BeforeDraw();
    Task_FractionAfterDraw *task_after  = getInterframe()->getTask_AfterDraw();
    Task_FractionCopy *task_copy        = getInterframe()->getTask_Copy();
    if(task_before->getOwner()==this)   { task_before->waitForComplete(); }
    if(task_after->getOwner()==this)    { task_after->waitForComplete(); }
    if(task_copy->getOwner()==this)     { task_copy->waitForComplete(); }
}



void FractionSet::updateSPH()
{
    SPHUpdateGrid();
    SPHComputeDensity();
    SPHComputeForce();
    SPHIntegrate();

    CUDA_SAFE_CALL( cudaMemcpyFromSymbol(m_particles, "d_particles", sizeof(m_particles), 0, cudaMemcpyDeviceToHost ) );
}

uint32 FractionSet::getNumBlocks() const
{
    const uint32 data_size = _countof(m_particles);
    return data_size/BLOCK_SIZE + (data_size%BLOCK_SIZE!=0 ? 1 : 0);
}

void FractionSet::setNext( FractionSet *next )
{
    m_next = next;
    if(next) {
        m_next->m_prev = this;
    }
}


void FractionSet::taskBeforeDraw()
{
    processMessage();
    updateSPH();
}

void FractionSet::taskBeforeDraw(uint32 block)
{
    move(block);
}

void FractionSet::taskAfterDraw()
{
    updateGrid();
}

void FractionSet::draw() const
{
    {
        Task_FractionBeforeDraw *task = getInterframe()->getTask_BeforeDraw();
        if(task->getOwner()==this) { TaskScheduler::waitExclusive(task); }
    }

    PassGBuffer_Cube *cube = atomicGetCubeRenderer();
    PassDeferred_SphereLight *light = atomicGetSphereLightRenderer();

    //cube->pushFractionInstance(make_float4(0.0f));
    light->pushInstance(make_float4(0.2f, 0.2f, 0.2f, 0.2f));

    size_t num_data = _countof(m_particles);
    for(uint32 i=0; i<num_data; ++i) {
        cube->pushFractionInstance(m_particles[i].position);
    }
    for(uint32 i=0; i<num_data; ++i) {
        if(m_particles[i].id % (SPH_MAX_PARTICLE_NUM>>6)==0) {
            light->pushInstance(m_particles[i].position);
        }
    }
}

void FractionSet::taskCopy( FractionSet *dst ) const
{
    dst->m_prev = this;
    dst->m_idgen = m_idgen;

    //uint32 num_data = _countof(m_data);
    //dst->m_data.clear();

    //const uint32 frame = atomicGetFrame();
    //for(uint32 i=0; i<num_data; ++i) {
    //    const FractionData& data = m_data[i];
    //    if(data.end_frame > frame) {
    //        dst->m_data.push_back(data);
    //    }
    //}
}



void FractionSet::processMessage()
{
    MessageIterator<Message_GenerateFraction> mes_gf_iter;
    while(mes_gf_iter.hasNext()) {
        const Message_GenerateFraction& mes = mes_gf_iter.iterate();
        for(uint32 n=0; n<mes.num; ++n) {
            if(mes.gen_type==Message_GenerateFraction::GEN_SPHERE) {
                ist::Sphere& sphere = (ist::Sphere&)(*mes.shape_data);
                FractionData fd;
                fd.id = ++m_idgen;
                fd.cell = 0;
                fd.end_frame = 0xFFFFFFFF;
                fd.density = 1.0f;
                fd.vel = _mm_set1_ps(0.0f);
                fd.pos = XMVectorSet(sphere.x, sphere.y, sphere.z, 0.0f);

                XMVECTOR r = atomicGenRandVector3();
                r = XMVectorSubtract(r, _mm_set1_ps(0.5f));
                r = XMVectorMultiply(r, _mm_set1_ps(2.0f));
                r = XMVectorMultiply(r, _mm_set1_ps(sphere.r));
                fd.pos = XMVectorAdd(fd.pos, r);
                XMVectorSetZ(fd.pos, 0.0f);
                fd.vel = XMVectorMultiply(atomicGenRandVector3(), _mm_set1_ps(0.2f));
                XMVectorSetZ(fd.vel, 0.0f);
                fd.accel = _mm_set1_ps(0.0f);

                //m_data.push_back(fd);
            }
            else if(mes.gen_type==Message_GenerateFraction::GEN_BOX) {
                IST_ASSERT("Message_GenerateFraction::GEN_BOX は未対応");
            }
        }
    }

    //const uint32 num_data = m_data.size();

    //getInterframe()->getGrid()->resizeData(num_data);
    //getInterframe()->resizeColliders(getNumBlocks());
}

void FractionSet::sphDensity(uint32 block)
{
    //if(!m_prev) { return; }
    //const uint32 num_data = m_data.size();
    //const uint32 begin = block*BLOCK_SIZE;
    //const uint32 end = std::min<uint32>((block+1)*BLOCK_SIZE, num_data);

    //const float32 h_sq = g_fSmoothlen*g_fSmoothlen;

    //for(uint32 i=begin; i<end; ++i) {
    //    FractionData& pdata = m_data[i];
    //    const XMVECTOR P_position = pdata.pos;
    //    float32 density = 1.0f;

    //    // 近傍グリッドを探査
    //    ivec3 index = GetIndexFromPositionHash(pdata.cell);
    //    for(uint32 zi=stl::max<uint32>(index.z-1, 0); zi<stl::min<uint32>(index.z+1, NUM_GRID_CELL); ++zi) {
    //        for(uint32 yi=stl::max<uint32>(index.y-1, 0); yi<stl::min<uint32>(index.y+1, NUM_GRID_CELL); ++yi) {
    //            for(uint32 xi=stl::max<uint32>(index.x-1, 0); xi<stl::min<uint32>(index.x+1, NUM_GRID_CELL); ++xi) {
    //                const FractionGridData &grid = m_prev->m_grid[GetHashFromIndex(xi, yi, zi)];
    //                for(uint32 j=grid.begin_index; j<grid.end_index; ++j) {
    //                    const FractionData& ndata = m_prev->m_data[j];
    //                    XMVECTOR N_position = ndata.pos;
    //                    XMVECTOR diff = XMVectorSubtract(N_position, P_position);
    //                    float32 r_sq = XMVectorGetX(XMVector3Dot(diff, diff));
    //                    if (r_sq < h_sq)
    //                    {
    //                        density += CalculateDensity(r_sq);
    //                    }
    //                }
    //            }
    //        }
    //    }

    //    pdata.density = density;
    //}
}


void FractionSet::sphForce(uint32 block)
{
    //if(!m_prev) { return; }
    //const uint32 num_data = m_data.size();
    //const uint32 begin = block*BLOCK_SIZE;
    //const uint32 end = std::min<uint32>((block+1)*BLOCK_SIZE, num_data);

    //const float32 h_sq = g_fSmoothlen*g_fSmoothlen;

    //for(uint32 i=begin; i<end; ++i) {
    //    FractionData& pdata = m_data[i];
    //    const XMVECTOR P_position = pdata.pos;
    //    const XMVECTOR P_velocity = pdata.vel;
    //    const float32 P_dencity = pdata.density;
    //    const float32 P_pressure = CalculatePressure(P_dencity);
    //    XMVECTOR acceleration = _mm_set1_ps(0.0f);

    //    // 近傍グリッドを探査
    //    ivec3 index = GetIndexFromPositionHash(pdata.cell);
    //    for(uint32 zi=stl::max<uint32>(index.z-1, 0); zi<stl::min<uint32>(index.z+1, NUM_GRID_CELL); ++zi) {
    //        for(uint32 yi=stl::max<uint32>(index.y-1, 0); yi<stl::min<uint32>(index.y+1, NUM_GRID_CELL); ++yi) {
    //            for(uint32 xi=stl::max<uint32>(index.x-1, 0); xi<stl::min<uint32>(index.x+1, NUM_GRID_CELL); ++xi) {
    //                const FractionGridData &grid = m_prev->m_grid[GetHashFromIndex(xi, yi, zi)];
    //                for(uint32 j=grid.begin_index; j<grid.end_index; ++j) {

    //                    const FractionData& ndata = m_prev->m_data[j];
    //                    if(pdata.id == ndata.id) { continue; }

    //                    XMVECTOR N_position = ndata.pos;
    //                    XMVECTOR diff = XMVectorSubtract(N_position, P_position);
    //                    float32 r_sq = XMVectorGetX(XMVector3Dot(diff, diff));
    //                    if(r_sq < h_sq) {
    //                        XMVECTOR N_velocity = ndata.vel;
    //                        float N_density = ndata.density;
    //                        float N_pressure = CalculatePressure(N_density);
    //                        float r = sqrt(r_sq);

    //                        // Pressure Term
    //                        acceleration += CalculateGradPressure(r, P_pressure, N_pressure, N_density, diff);

    //                        // Viscosity Term
    //                        acceleration += CalculateLapVelocity(r, P_velocity, N_velocity, N_density);
    //                    }
    //                }
    //            }
    //        }
    //    }
    //    pdata.accel = XMVectorMultiply(acceleration, _mm_set1_ps(1.0f/P_dencity));
    //}

}

void FractionSet::move(uint32 block)
{
    //const uint32 num_data = m_data.size();
    //const uint32 begin = block*BLOCK_SIZE;
    //const uint32 end = std::min<uint32>((block+1)*BLOCK_SIZE, num_data);

    //const float32 GRAVITY = 0.5f;
    //const float SPHERE_RADIUS = 0.5f;
    //const float OUTER_SPHERE_RADIUS = 2.56f;
    //XMVECTOR gravity_center     = XMVectorSet(0.0f, 0.0f, 0.0f, 0.0f);
    //XMVECTOR gravity_strength   = _mm_set1_ps(GRAVITY);
    //XMVECTOR max_speed          = _mm_set1_ps(FractionSet::MAX_VEL);
    //XMVECTOR decelerate         = _mm_set1_ps(FractionSet::DECELERATE);
    //XMVECTOR sphere_radius      = _mm_set1_ps(SPHERE_RADIUS);
    //XMVECTOR timestep           = _mm_set1_ps(1.0f/60.0f);

    //for(uint32 i=begin; i<end; ++i) {
    //    FractionData &data = m_data[i];

    //    XMVECTOR pos        = data.pos;
    //    XMVECTOR vel        = data.vel;
    //    XMVECTOR accel      = data.accel;

    //    XMVECTOR g_dist     = XMVectorSubtract(gravity_center, pos);
    //    XMVECTOR g_dir      = XMVector3Normalize(g_dist);
    //    XMVECTOR g_accel    = XMVectorMultiply(g_dir, gravity_strength); // 重力加速
    //    //XMVECTOR g_accel    = XMVectorSet(0.0f, -GRAVITY, 0.0f, 0.0f); // 重力加速

    //    accel               = XMVectorAdd(accel, g_accel);
    //    vel                 = XMVectorAdd(vel, XMVectorMultiply(accel, timestep));
    //    XMVECTOR v_len      = XMVector3Length(vel);
    //    XMVECTOR v_dec      = XMVectorMultiply(v_len, decelerate);
    //    XMVECTOR v_select   = XMVectorGreater(v_len, max_speed);
    //    XMVECTOR v_speed    = XMVectorSelect(v_len, v_dec, v_select);
    //    XMVECTOR v_dir      = XMVectorDivide(vel, v_len);
    //    XMVECTOR v_next     = XMVectorMultiply(v_dir, v_speed); // 重力加速後の速度
    //    v_next = XMVectorMultiply(v_next, XMVectorSet(1.0f, 1.0f, 0.2f, 1.0f));

    //    // 最高速度超えてたら減速

    //    pos                 = XMVectorAdd(pos, XMVectorMultiply(v_next, timestep));
    //    XMVECTOR p_dist     = XMVectorSubtract(pos, gravity_center);
    //    XMVECTOR p_len      = XMVector3Length(p_dist);
    //    XMVECTOR is_bound   = XMVectorLess(p_len, sphere_radius); // 跳ね返ったか
    //    XMVECTOR p_dir      = XMVectorDivide(p_dist, p_len);
    //    p_dir = XMVectorMultiply(p_dir, XMVectorSet(1.0f, 1.0f, 1.0f, 0.0f));

    //    XMVECTOR is_outer_bound = XMVectorGreater(p_len, _mm_set1_ps(OUTER_SPHERE_RADIUS));


    //    pos = XMVectorMultiply(pos, XMVectorSet(1.0f, 1.0f, 0.2f, 1.0f));

    //    // 跳ね返ってるのであれば速度を反転
    //    XMVECTOR r_pos;
    //    XMVECTOR r_vel;
    //    if(XMVectorGetIntX(is_bound)) {
    //        if(XMVectorGetX(v_speed)<=GRAVITY) {
    //            r_vel = _mm_set1_ps(0.0f);
    //        }
    //        else {
    //            XMMATRIX reflection = XMMatrixReflect(p_dir);
    //            XMVECTOR tv = XMVector3Transform(v_next, reflection);
    //            r_vel = XMVectorMultiply(tv, _mm_set1_ps(BOUNCE));
    //        }
    //        float d = SPHERE_RADIUS;
    //        r_pos = XMVectorMultiply(p_dir, _mm_set1_ps(d));
    //    }
    //    else if(XMVectorGetIntX(is_outer_bound)) {
    //        r_vel = XMVectorMultiply(XMVectorNegate(r_vel), _mm_set1_ps(BOUNCE));
    //        r_pos = XMVectorMultiply(p_dir, _mm_set1_ps(OUTER_SPHERE_RADIUS));
    //    }
    //    else {
    //        r_pos = pos;
    //        r_vel = v_next;
    //    }

    //    data.pos = r_pos;
    //    data.vel = r_vel;
    //}
}

struct less_pos_hash
{
    bool operator()(const FractionData& lhs, const FractionData& rhs)
    {
        return lhs.cell < rhs.cell;
    }
};

void FractionSet::updateGrid()
{
    //const uint32 data_size = m_data.size();
    //for(uint32 i=0; i<data_size; ++i) {
    //    m_data[i].cell = GenerateHashFromPosition(m_data[i].pos);
    //}
    //stl::sort(m_data.begin(), m_data.end(), less_pos_hash());

    //{
    //    const uint32 grid_size = m_grid.size();
    //    for(uint32 i=0; i<grid_size; ++i) {
    //        FractionGridData& fgd = m_grid[i];
    //        fgd.begin_index = fgd.end_index = 0;
    //    }
    //}

    //{
    //    uint32 prev_hash = 0;
    //    for(uint32 i=0; i<data_size; ++i) {
    //        const uint32 id      = i;
    //        const uint32 id_prev = i==0 ? 0 : i-1;
    //        const uint32 id_next = i+1==data_size ? 0 : i+1;

    //        const uint32 cell      = m_data[id].cell;
    //        const uint32 cell_prev = m_data[id_prev].cell;
    //        const uint32 cell_next = m_data[id_next].cell;
    //        if(cell != cell_prev) {
    //            m_grid[cell].begin_index = i;
    //        }
    //        if(cell != cell_next) {
    //            m_grid[cell].end_index = i;
    //        }
    //    }
    //}
}

} // namespace atomic
