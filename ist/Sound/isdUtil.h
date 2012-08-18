#ifndef __ist_isd_Util__
#define __ist_isd_Util__

#include "isdTypes.h"

namespace ist {
namespace isd {

istInterModule bool CreateBufferFromWaveFile(const char *filepath, Buffer *buf);
istInterModule bool CreateBufferFromOggFile(const char *filepath, Buffer *buf);

istInterModule Stream* CreateStreamFromWaveFile(const char* filepath);
istInterModule Stream* CreateStreamFromOggFile(const char* filepath);


} // namespace isd
} // namespace ist

#endif // __ist_isd_Util__
