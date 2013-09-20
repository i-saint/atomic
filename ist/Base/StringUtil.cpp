#include "istPCH.h"
#include "StringUtil.h"

namespace ist {

istAPI size_t scan(const std::string &content, const std::regex &pattern, const std::function<void (const std::cmatch &m)> &f)
{
    std::cmatch m;
    size_t num = 0;
    size_t pos = 0;
    while(std::regex_search(content.c_str()+pos, m, pattern)) {
        f(m);
        pos += m.position()+m.length();
        ++num;
    }
    return num;
}

istAPI std::string sub(const std::string &content, const std::regex &pattern, const std::function<std::string (const std::cmatch &m)> &f)
{
    struct replace_info {
        size_t pos, len;
        std::string replacement;
    };
    std::vector<replace_info> replacements;
    std::cmatch m;
    if(std::regex_search(content.c_str(), m, pattern)) {
        replace_info ri = {m.position(), m.length(), f(m)};
        replacements.push_back(ri);
    }

    std::string ret = content;
    std::for_each(replacements.rbegin(), replacements.rend(), [&](replace_info &ri){
        ret.replace(ri.pos, ri.len, ri.replacement);
    });
    return ret;
}

istAPI std::string gsub(const std::string &content, const std::regex &pattern, const std::function<std::string (const std::cmatch &m)> &f)
{
    struct replace_info {
        size_t pos, len;
        std::string replacement;
    };
    std::vector<replace_info> replacements;
    std::cmatch m;
    size_t pos = 0;
    while(std::regex_search(content.c_str()+pos, m, pattern)) {
        replace_info ri = {pos+m.position(), m.length(), f(m)};
        replacements.push_back(ri);
        pos += m.position()+m.length();
    }

    std::string ret = content;
    std::for_each(replacements.rbegin(), replacements.rend(), [&](replace_info &ri){
        ret.replace(ri.pos, ri.len, ri.replacement);
    });
    return ret;
}

} // namespace ist
