#ifndef ist_Base_FileIO_h
#define ist_Base_FileIO_h


namespace ist {

bool FileToString(const char *path, stl::string &out);
bool FileToString(const stl::string &path, stl::string &out);

} // namespace ist

#endif // ist_Base_FileIO_h
