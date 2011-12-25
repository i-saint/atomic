

struct DeviceForceDataSet
{
    sphForcePointGravity    *point_gravity;
    int                     num_point_gravity;
};


struct ForceDataSet
{
    thrust::device_vector<sphForcePointGravity> point_gravity;

    ForceDataSet()
    {
        point_gravity.reserve(SPH_MAX_POINT_GRAVITY_NUM);
    }

    DeviceForceDataSet getDeviceData()
    {
        DeviceForceDataSet ddata;
        ddata.point_gravity     = point_gravity.data().get();
        ddata.num_point_gravity = point_gravity.size();
        return ddata;
    }
};
