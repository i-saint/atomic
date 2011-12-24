
struct DeviceRigidDataSet
{
    sphGridParam            *params;
    sphStates               *states;
    sphRigidClass           *classinfo;
    sphRigidInstance        *instances;
    sphRigidParticle        *particles;
    sphRigidUpdateInfo      *updateinfo;
    sphHash                 *hashes;
    sphGridData             *grid;

    __device__ int3 GridCalculateCell(float4 pos)
    {
        float4 c = (pos-params->grid_pos)*params->grid_dim_rcp;
        int3 uc = make_int3(c.x, c.y, c.z);
        return clamp(uc, make_int3(0), make_int3(SPH_RIGID_GRID_DIV_X-1, SPH_RIGID_GRID_DIV_Y-1, SPH_RIGID_GRID_DIV_Z-1));
    }

    __device__ uint GridCalculateHash(float4 pos)
    {
        return GridConstuctKey( GridCalculateCell(pos) );
    }

    __device__ uint GridConstuctKey(int3 v)
    {
        return v.x | (v.y<<SPH_RIGID_GRID_DIV_SHIFT_X) | (v.z<<(SPH_RIGID_GRID_DIV_SHIFT_X+SPH_RIGID_GRID_DIV_SHIFT_Y));
    }


    __device__ void updateHash(int i)
    {
        hashes[i] = GridCalculateHash(particles[i].position);
    }

    __device__ void clearGrid(int i)
    {
        grid[i].x = grid[i].y = 0;
    }

    __device__ void updateGrid(int i)
    {
        const uint G_ID = i;
        uint G_ID_PREV = (G_ID == 0)? SPH_MAX_RIGID_PARTICLES : G_ID; G_ID_PREV--;
        uint G_ID_NEXT = G_ID + 1; if (G_ID_NEXT == SPH_MAX_RIGID_PARTICLES) { G_ID_NEXT = 0; }
    
        uint cell = hashes[G_ID];
        uint cell_prev = hashes[G_ID_PREV];
        uint cell_next = hashes[G_ID_NEXT];
        if (cell != cell_prev)
        {
            // I'm the start of a cell
            grid[cell].x = G_ID;
        }
        if (cell != cell_next)
        {
            // I'm the end of a cell
            grid[cell].y = G_ID + 1;
        }
    }

    __device__ void updateRigids(int i)
    {
        sphRigidUpdateInfo  &rui    = updateinfo[i];
        sphRigidClass       &rc     = classinfo[rui.classid];
        sphRigidInstance    &rin    = instances[rui.cindex];
        sphRigidParticle    &rp     = rc.particles[rui.pindex];
        particles[i].owner_handle   = rui.owner_handle;
        particles[i].position       = vector_cast<float4&>(rin.transform * vector_cast<vec4>(rp.position));
        particles[i].normal         = vector_cast<float4&>(rin.transform * vector_cast<vec4>(rp.normal));
        particles[i].padding.y      = 0.0f;
        particles[i].padding.z      = 0.0f;
    }
};

struct RigidDataSet
{
    thrust::device_vector<sphGridParam>         params;
    thrust::device_vector<sphStates>            states;
    thrust::device_vector<sphRigidClass>        classinfo;
    thrust::device_vector<sphRigidInstance>     instances;
    thrust::device_vector<sphRigidParticle>     particles;
    thrust::device_vector<sphRigidUpdateInfo>   updateinfo;
    thrust::device_vector<sphHash>              hashes;
    thrust::device_vector<sphGridData>          grid;

    RigidDataSet()
    {
        params.resize(1);
        states.resize(1);
        classinfo.resize(atomic::CB_END);
        instances.reserve(atomic::ATOMIC_MAX_CHARACTERS);
        particles.reserve(SPH_MAX_RIGID_PARTICLES);
        updateinfo.reserve(SPH_MAX_RIGID_PARTICLES);
        hashes.reserve(SPH_MAX_RIGID_PARTICLES);
        grid.resize(SPH_RIGID_GRID_DIV_3);
    }

    void resizeParticles(size_t n)
    {
        particles.resize(n);
        hashes.resize(n);
    }

    DeviceRigidDataSet getDeviceData()
    {
        DeviceRigidDataSet ddata;
        ddata.params    = params.data().get();
        ddata.states    = states.data().get();
        ddata.classinfo = classinfo.data().get();
        ddata.instances = instances.data().get();
        ddata.particles = particles.data().get();
        ddata.updateinfo= updateinfo.data().get();
        ddata.hashes    = hashes.data().get();
        ddata.grid      = grid.data().get();
        return ddata;
    }
};

struct _RigidUpdate
{
    DeviceRigidDataSet drd;
    _RigidUpdate(const DeviceRigidDataSet &v) : drd(v) {}
    __device__ void operator()(int i)
    {
        drd.updateRigids(i);
    }
};

struct _RigidUpdateHash
{
    DeviceRigidDataSet drd;
    _RigidUpdateHash(const DeviceRigidDataSet &v) : drd(v) {}
    __device__ void operator()(int i) { drd.updateHash(i); }
};

struct _RigidClearGrid
{
    DeviceRigidDataSet drd;
    _RigidClearGrid(const DeviceRigidDataSet &v) : drd(v) {}
    __device__ void operator()(int i) { drd.clearGrid(i); }
};

struct _RigidUpdateGrid
{
    DeviceRigidDataSet drd;
    _RigidUpdateGrid(const DeviceRigidDataSet  &v) : drd(v) {}
    __device__ void operator()(int i) { drd.updateGrid(i); }
};
