
struct DeviceRigidDataSet
{
    sphRigidSphere  *spheres;
    sphRigidBox     *boxes;
    int             num_spheres;
    int             num_boxes;
};

struct RigidDataSet
{
    thrust::device_vector<sphRigidSphere>       spheres;
    thrust::device_vector<sphRigidBox>          boxes;

    RigidDataSet()
    {
        spheres.reserve(atomic::ATOMIC_MAX_CHARACTERS);
        boxes.reserve(atomic::ATOMIC_MAX_CHARACTERS);
    }

    DeviceRigidDataSet getDeviceData()
    {
        DeviceRigidDataSet ddata;
        ddata.spheres       = spheres.data().get();
        ddata.boxes         = boxes.data().get();
        ddata.num_spheres   = spheres.size();
        ddata.num_boxes     = boxes.size();
        return ddata;
    }
};
