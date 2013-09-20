#ifndef ist_Debug_EachMembers_h
#define ist_Debug_EachMembers_h
#include <functional>
#include <string>
#include <vector>

namespace ist {

enum MemberType {
    MT_Unknown,
    MT_Variable,
    MT_BaseClass,
    MT_Function,
};

struct MemberInfo
{
    MemberType type;
    void *this_pointer;     // this
    void *base_pointer;     // 親 class のメンバを指してる場合、親 class の先頭を指す
    void *value;            // メンバ変数へのポインタ
    std::string this_type;  // this の型
    std::string class_name; // メンバ変数が所属する class。親 class のメンバの場合 this_type とは違うものになる
    std::string type_name;  // メンバ変数の型名
    std::string value_name; // メンバ変数の名前

    MemberInfo() : type(MT_Unknown), this_pointer(), base_pointer(), value() {}
};
typedef std::function<void (const MemberInfo*, size_t)> MemberInfoCallback;

istAPI bool EachMembersByTypeName(const char *classname, const MemberInfoCallback &f);
istAPI bool EachMembersByTypeName(const char *classname, void *_this, const MemberInfoCallback &f);
istAPI bool EachMembersByPointer(void *_this, const MemberInfoCallback &f);

} // namespace ist
#endif // ist_Debug_EachMembers_h
