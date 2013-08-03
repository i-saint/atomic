#ifndef __ist_isd_Util__
#define __ist_isd_Util__

#include "isdTypes.h"

namespace ist {
namespace isd {

istAPI bool CreateBufferFromWaveFile(const char *filepath, Buffer *buf);
istAPI bool CreateBufferFromOggFile(const char *filepath, Buffer *buf);

istAPI Stream* CreateStreamFromWaveFile(const char* filepath);
istAPI Stream* CreateStreamFromOggFile(const char* filepath);


} // namespace isd
} // namespace ist

#endif // __ist_isd_Util__
