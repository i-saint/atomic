#ifndef ist_Base_FileIO_h
#define ist_Base_FileIO_h

#include "../Config.h"

namespace ist {

istAPI bool FileToString(const char *path, stl::string &out);
istAPI bool FileToString(const stl::string &path, stl::string &out);

} // namespace ist

#endif // ist_Base_FileIO_h
