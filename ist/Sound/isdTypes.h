#ifndef __ist_isd_Types__
#define __ist_isd_Types__

#include "ist/Base/New.h"

namespace ist {
namespace isd {

typedef uint32 ResourceHandle;
class istInterModule Device;
class istInterModule DeviceResource;
class istInterModule Buffer;
class istInterModule Source;
class istInterModule Listener;

class istInterModule Stream;

} // namespace isd
} // namespace ist

#define ISD_DECLARE_DEVICE_RESOURCE(classname) \
private:\
    istMakeDestructable;\
    friend class Device;\
    friend class DeviceContext;

#endif // __ist_isd_Types__
