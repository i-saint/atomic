#ifndef ist_Base_StringUtil_h
#define ist_Base_StringUtil_h
#include "ist/Config.h"
#include <string>
#include <regex>
#include <functional>

namespace ist {

istAPI size_t scan(const std::string &content, const std::regex &pattern, const std::function<void (const std::cmatch &m)> &f);
istAPI std::string sub(const std::string &content, const std::regex &pattern, const std::function<std::string (const std::cmatch &m)> &f);
istAPI std::string gsub(const std::string &content, const std::regex &pattern, const std::function<std::string (const std::cmatch &m)> &f);

} // namespace ist
#endif // ist_Base_StringUtil_h
