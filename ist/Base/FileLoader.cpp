#include "istPCH.h"
#include "FileLoader.h"
#include <fstream>

namespace ist {

bool ist::FileToString( const char *path, stl::string &out )
{
    std::fstream fin(path, std::ios::in | std::ios::binary);
    if(!fin) { return false; }
    fin.seekg(0, std::ios::end);
    out.resize((size_t)fin.tellg());
    fin.seekg(0, std::ios::beg);
    fin.read(&out[0], out.size());
    fin.close();
    return true;
}

bool FileToString( const stl::string &path, stl::string &out )
{
    return FileToString(path.c_str(), out);
}

} // namespace ist

