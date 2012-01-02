#ifndef __ist_isd_Util__
#define __ist_isd_Util__

#include "isdTypes.h"

namespace ist {
namespace isd {

bool CreateBufferFromWaveFile(const char *filepath, Buffer *buf);
bool CreateBufferFromOggFile(const char *filepath, Buffer *buf);

Stream* CreateStreamFromWaveFile(const char* filepath);
Stream* CreateStreamFromOggFile(const char* filepath);


class ReferenceCounter
{
private:
    uint32 m_reference_count;

public:
    ReferenceCounter();
    uint32 getRef() const;
    uint32 addRef();
    uint32 release();
};

} // namespace isd
} // namespace ist

#endif // __ist_isd_Util__
