#include "istPCH.h"
#include "FileLoader.h"

namespace ist {

bool ist::FileToString( const char *path, stl::string &out )
{
    FILE *fin = fopen(path, "rb");
    if(!fin) { return false; }
    fseek(fin, 0, SEEK_END);
    out.resize((size_t)ftell(fin));
    fseek(fin, 0, SEEK_SET);
    fread(&out[0], 1, out.size(), fin);
    fclose(fin);
    return true;
}

bool FileToString( const stl::string &path, stl::string &out )
{
    return FileToString(path.c_str(), out);
}

} // namespace ist

