#ifndef __ist_isd_Listener__
#define __ist_isd_Listener__

#include "isdDeviceResource.h"

namespace ist {
namespace isd {


class istInterModule Listener : public DeviceResource
{
ISD_DECLARE_DEVICE_RESOURCE(Listener);
typedef DeviceResource super;
private:

protected:
    int     getI(ALenum param) const;
    float   getF(ALenum param) const;
    vec3    get3F(ALenum param) const;
    void setI(ALenum param, int v);
    void setF(ALenum param, float v);
    void set3F(ALenum param, const vec3& v);

    Listener(Device *dev);
    virtual ~Listener();

public:
    float getGain() const;
    vec3 getPosition() const;
    vec3 getVelocity() const;

    void setGain(float v);
    void setPosition(const vec3& v);
    void setVelocity(const vec3& v);
};

} // namespace isd
} // namespace ist

#endif // __ist_isd_Listener__
