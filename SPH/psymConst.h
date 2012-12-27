#ifndef _SPH_const_h_
#define _SPH_const_h_

#define PSYM_PARTICLE_SIZE 0.08f

//#define PSYM_MAX_PARTICLE_NUM 100000

#define PSYM_MAX_PARTICLE_NUM 131072
//#define PSYM_MAX_PARTICLE_NUM 65536
//#define PSYM_MAX_PARTICLE_NUM 32768
//#define PSYM_MAX_PARTICLE_NUM 16384
//#define PSYM_MAX_PARTICLE_NUM 1024

#define PSYM_GRID_SIZE 10.24f
#define PSYM_GRID_POS -10.24f
#define PSYM_GRID_CELL_SIZE 0.08f
#define PSYM_GRID_DIV 256
#define PSYM_GRID_DIV_BITS 8
#define PSYM_GRID_CELL_NUM (PSYM_GRID_DIV*PSYM_GRID_DIV)

#define PSYM_WALL_STIFFNESS 300.0f


//#define SPH_enable_neighbor_density_estimation


#endif // _SPH_const_h_
