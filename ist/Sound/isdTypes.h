#ifndef __ist_isd_Types__
#define __ist_isd_Types__

#include "ist/Base/New.h"

namespace ist {
namespace isd {

typedef uint32 ResourceHandle;
class Device;
class DeviceResource;
class Buffer;
class Source;
class Listener;

class Stream;

} // namespace isd
} // namespace ist

#define ISD_DECLARE_DEVICE_RESOURCE(classname) \
private:\
    template<class T> friend T* ::call_destructor(T *v);\
    friend class Device;\
    friend class DeviceContext;

#endif // __ist_isd_Types__
