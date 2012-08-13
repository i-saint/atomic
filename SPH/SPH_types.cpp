//#include "stdafx.h"
#include "SPH_types.h"
#include <tbb/tbb.h>
#include <algorithm>

const int32 SIMD_LANES = 8;

template<class T>
T clamp(T v, T minv, T maxv)
{
    return std::min<T>(std::max<T>(v, minv), maxv);
}

int32 soa_blocks(int32 i)
{
    return (i/SIMD_LANES) + (i%SIMD_LANES>0 ? 1 : 0);
}

void SoAnize( int32 num, const sphParticle *particles, ispc::Particle_SOA8 *out )
{
    int32 blocks = soa_blocks(num);
    for(int32 bi=0; bi<blocks; ++bi) {
        int32 i = SIMD_LANES*bi;
        ist::soavec34 soav;
        simdvec4 soas;
        soav = ist::soa_transpose34(particles[i+0].position, particles[i+1].position, particles[i+2].position, particles[i+3].position);
        _mm_store_ps(out[bi].x+0, soav.x());
        _mm_store_ps(out[bi].y+0, soav.y());
        _mm_store_ps(out[bi].z+0, soav.z());
        soav = ist::soa_transpose34(particles[i+0].velocity, particles[i+1].velocity, particles[i+2].velocity, particles[i+3].velocity);
        _mm_store_ps(out[bi].vx+0, soav.x());
        _mm_store_ps(out[bi].vy+0, soav.y());
        _mm_store_ps(out[bi].vz+0, soav.z());

        soav = ist::soa_transpose34(particles[i+4].position, particles[i+5].position, particles[i+6].position, particles[i+7].position);
        _mm_store_ps(out[bi].x+4, soav.x());
        _mm_store_ps(out[bi].y+4, soav.y());
        _mm_store_ps(out[bi].z+4, soav.z());
        soav = ist::soa_transpose34(particles[i+4].velocity, particles[i+5].velocity, particles[i+6].velocity, particles[i+7].velocity);
        _mm_store_ps(out[bi].vx+4, soav.x());
        _mm_store_ps(out[bi].vy+4, soav.y());
        _mm_store_ps(out[bi].vz+4, soav.z());

        soas = ist::simdvec4_set(particles[i+0].params.density, particles[i+1].params.density, particles[i+2].params.density, particles[i+3].params.density);
        _mm_store_ps(out[bi].density+0, soas);
        soas = ist::simdvec4_set(particles[i+4].params.density, particles[i+5].params.density, particles[i+6].params.density, particles[i+7].params.density);
        _mm_store_ps(out[bi].density+4, soas);
    }
}

void AoSnize( int32 num, const ispc::Particle_SOA8 *particles, sphParticle *out )
{
    int32 blocks = soa_blocks(num);
    for(int32 bi=0; bi<blocks; ++bi) {
        int32 i = 8*bi;
        ist::soavec44 aos_pos[2] = {
            ist::soa_transpose44(
                _mm_load_ps(particles[bi].x + 0),
                _mm_load_ps(particles[bi].y + 0),
                _mm_load_ps(particles[bi].z + 0),
                _mm_set1_ps(1.0f) ),
            ist::soa_transpose44(
                _mm_load_ps(particles[bi].x + 4),
                _mm_load_ps(particles[bi].y + 4),
                _mm_load_ps(particles[bi].z + 4),
                _mm_set1_ps(1.0f) ),
        };
        ist::soavec44 aos_vel[2] = {
            ist::soa_transpose44(
                _mm_load_ps(particles[bi].vx + 0),
                _mm_load_ps(particles[bi].vy + 0),
                _mm_load_ps(particles[bi].vz + 0),
                _mm_set1_ps(0.0f) ),
            ist::soa_transpose44(
                _mm_load_ps(particles[bi].vx + 4),
                _mm_load_ps(particles[bi].vy + 4),
                _mm_load_ps(particles[bi].vz + 4),
                _mm_set1_ps(0.0f) ),
        };

        int32 e = std::min<int32>(SIMD_LANES, num-i);
        for(int32 ei=0; ei<e; ++ei) {
            out[i+ei].position = aos_pos[ei/4][ei%4];
            out[i+ei].velocity = aos_vel[ei/4][ei%4];
            out[i+ei].params.density = particles[bi].density[ei];
        }
    }
}

inline int32 GenHash(const float *pos4)
{
    static const float32 rcpcellsize = 1.0f/SPH_GRID_CELL_SIZE;
    int32 r=(clamp<int32>(int32((pos4[0]-SPH_GRID_POS)*rcpcellsize), 0, (SPH_GRID_DIV-1)) << (SPH_GRID_DIV_BITS*0)) |
            (clamp<int32>(int32((pos4[1]-SPH_GRID_POS)*rcpcellsize), 0, (SPH_GRID_DIV-1)) << (SPH_GRID_DIV_BITS*1));
    return r;
}

inline void GenIndex(uint32 hash, int32 &xi, int32 &yi)
{
    xi = (hash >> (SPH_GRID_DIV_BITS*0)) & (SPH_GRID_DIV-1);
    yi = (hash >> (SPH_GRID_DIV_BITS*1)) & (SPH_GRID_DIV-1);
}

sphGrid::sphGrid()
{
    for(uint32 i=0; i<_countof(particles); ++i) {
        particles[i].position = ist::simdvec4_set(
            SPH_PARTICLE_SIZE*0.5f * (i % (SPH_GRID_DIV*2)) - SPH_GRID_SIZE + 0.00010f*i,
            SPH_PARTICLE_SIZE*0.5f * (i / (SPH_GRID_DIV*4)) - 0.0f + 0.00011f*i,
            SPH_PARTICLE_SIZE*0.5f * (i / (SPH_GRID_DIV*1)) + SPH_GRID_SIZE - 0.00012f*i,
            //0.0f,
            1.0f );
        particles[i].velocity = ist::simdvec4_set(0.0f, 0.0f, 0.0f, 0.0f);
        particles[i].params.density = 0.0f;
    }
}

void sphGrid::update()
{
    sphParticle *ps = particles;            // メンバ変数は lampda 関数に取り込めないのでローカルに落とす
    sphGridData *ce = &cell[0][0];          // 
    ispc::Particle_SOA8 *so = particles_soa;// 

    // clear grid
    tbb::parallel_for(tbb::blocked_range<int>(0, SPH_GRID_CELL_NUM, 128),
        [=](const tbb::blocked_range<int> &r) {
            for(int i=r.begin(); i!=r.end(); ++i) {
                ce[i].begin = ce[i].end = 0;
            }
        });

    // gen hash
    tbb::parallel_for(tbb::blocked_range<int>(0, SPH_MAX_PARTICLE_NUM, 1024),
        [=](const tbb::blocked_range<int> &r) {
            for(int i=r.begin(); i!=r.end(); ++i) {
                ps[i].params.hash = GenHash((float*)&ps[i].position);
            }
        });

    // パーティクルを hash で sort
    tbb::parallel_sort(particles, particles+SPH_MAX_PARTICLE_NUM, 
        [=](const sphParticle &a, const sphParticle &b) { return a.params.hash < b.params.hash; } );

    // パーティクルがどの grid に入っているかを算出
    tbb::parallel_for(tbb::blocked_range<int>(0, SPH_MAX_PARTICLE_NUM, 1024),
        [=](const tbb::blocked_range<int> &r) {
            for(int i=r.begin(); i!=r.end(); ++i) {
                const int32 G_ID = i;
                int32 G_ID_PREV = G_ID-1;
                int32 G_ID_NEXT = G_ID+1;

                int32 cell = ps[G_ID].params.hash;
                int32 cell_prev = (G_ID_PREV==-1) ? -1 : ps[G_ID_PREV].params.hash;
                int32 cell_next = (G_ID_NEXT==SPH_MAX_PARTICLE_NUM) ? -2 : ps[G_ID_NEXT].params.hash;
                if(cell != cell_prev) {
                    ce[cell].begin = G_ID;
                }
                if(cell != cell_next) {
                    ce[cell].end = G_ID + 1;
                }
            }
    });

    {
        int32 soai = 0;
        for(int i=0; i!=SPH_GRID_CELL_NUM; ++i) {
            ce[i].soai = soai;
            soai += soa_blocks(ce[i].end-ce[i].begin);
        }
    }

    // AoS -> SoA
    tbb::parallel_for(tbb::blocked_range<int>(0, SPH_GRID_CELL_NUM, 128),
        [=](const tbb::blocked_range<int> &r) {
            for(int i=r.begin(); i!=r.end(); ++i) {
                int32 n = ce[i].end - ce[i].begin;
                if(n > 0) {
                    sphParticle *p = &ps[ce[i].begin];
                    ispc::Particle_SOA8 *t = &so[ce[i].soai];
                    SoAnize(n, p, t);
                }
            }
    });

    // SPH
    tbb::parallel_for(tbb::blocked_range<int>(0, SPH_GRID_CELL_NUM, 128),
        [=](const tbb::blocked_range<int> &r) {
            for(int i=r.begin(); i!=r.end(); ++i) {
                int32 n = ce[i].end - ce[i].begin;
                if(n > 0) {
                    int xi, yi;
                    GenIndex(i, xi, yi);
                    ispc::sphUpdateDensity((ispc::Particle*)so, ce, xi, yi);
                }
            }
    });
    tbb::parallel_for(tbb::blocked_range<int>(0, SPH_GRID_CELL_NUM, 128),
        [=](const tbb::blocked_range<int> &r) {
            for(int i=r.begin(); i!=r.end(); ++i) {
                int32 n = ce[i].end - ce[i].begin;
                if(n > 0) {
                    int xi, yi;
                    GenIndex(i, xi, yi);
                    ispc::sphUpdateForce((ispc::Particle*)so, ce, xi, yi);
                }
            }
    });
    tbb::parallel_for(tbb::blocked_range<int>(0, SPH_GRID_CELL_NUM, 128),
        [=](const tbb::blocked_range<int> &r) {
            for(int i=r.begin(); i!=r.end(); ++i) {
                int32 n = ce[i].end - ce[i].begin;
                if(n > 0) {
                    int xi, yi;
                    GenIndex(i, xi, yi);
                    ispc::sphIntegrate((ispc::Particle*)so, ce, xi, yi);
                }
            }
    });

    //// impulse
    //tbb::parallel_for(tbb::blocked_range<int>(0, SPH_GRID_CELL_NUM, 128),
    //    [=](const tbb::blocked_range<int> &r) {
    //        for(int i=r.begin(); i!=r.end(); ++i) {
    //            int32 n = ce[i].end - ce[i].begin;
    //            if(n > 0) {
    //                int xi, yi;
    //                GenIndex(i, xi, yi);
    //                ispc::impUpdateVelocity((ispc::Particle*)so, ce, xi, yi);
    //            }
    //        }
    //});
    //tbb::parallel_for(tbb::blocked_range<int>(0, SPH_GRID_CELL_NUM, 128),
    //    [=](const tbb::blocked_range<int> &r) {
    //        for(int i=r.begin(); i!=r.end(); ++i) {
    //            int32 n = ce[i].end - ce[i].begin;
    //            if(n > 0) {
    //                int xi, yi;
    //                GenIndex(i, xi, yi);
    //                ispc::impIntegrate((ispc::Particle*)so, ce, xi, yi);
    //            }
    //        }
    //});

    // SoA -> AoS
    tbb::parallel_for(tbb::blocked_range<int>(0, SPH_GRID_CELL_NUM, 128),
        [=](const tbb::blocked_range<int> &r) {
            for(int i=r.begin(); i!=r.end(); ++i) {
                int32 n = ce[i].end - ce[i].begin;
                if(n > 0) {
                    sphParticle *p = &ps[ce[i].begin];
                    ispc::Particle_SOA8 *t = &so[ce[i].soai];
                    AoSnize(n, t, p);
                }
            }
    });
}
