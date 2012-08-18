#ifndef __ist_isd_Buffer__
#define __ist_isd_Buffer__

#include "isdDeviceResource.h"

namespace ist {
namespace isd {


class istInterModule Buffer : public DeviceResource
{
ISD_DECLARE_DEVICE_RESOURCE(Buffer);
private:
    void initialize();

protected:
    int getI(ALenum param) const;

    Buffer();
    virtual ~Buffer();

public:

    int getSize() const;
    int getBits() const;
    int getChannels() const;
    int getFrequency() const;

    /// format: AL_FORMAT_MONO8  AL_FORMAT_MONO16 AL_FORMAT_STEREO8 AL_FORMAT_STEREO16 
    void copy(char *data, size_t size, ALenum format, int samplerate);
};

} // namespace isd
} // namespace ist

#endif // __ist_isd_Buffer__
