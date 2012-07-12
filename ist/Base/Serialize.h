#ifndef __ist_Base_Serialize__
#define __ist_Base_Serialize__

#include <cstdlib>
#include <vector>
#include <algorithm>

namespace ist {

#define IST_INTROSPECTION_INTERFACE(type) \
    typedef type self_t; \
    static const ist::MemberInfoCollection& _GetMemberInfo() { \
        static ist::MemberInfoCollection collection(NULL, 0);\
        return collection;\
    }\
    virtual const ist::MemberInfoCollection& GetMemberInfo() { return _GetMemberInfo(); }

#define IST_INTROSPECTION(type, members) \
    typedef type self_t; \
    static const ist::MemberInfoCollection& _GetMemberInfo() { \
        static ist::IMemberInfo *data[] = { members }; \
        static ist::MemberInfoCollection collection(data, _countof(data));\
        return collection;\
    }\
    virtual const ist::MemberInfoCollection& GetMemberInfo() { return _GetMemberInfo(); }

#define IST_INTROSPECTION_INHERIT(type, supers, members) \
    typedef type self_t; \
    static const ist::MemberInfoCollection& _GetMemberInfo() { \
        static ist::MemberInfoCollection s[] = { supers }; \
        static ist::IMemberInfo *data[] = { members }; \
        static ist::MemberInfoCollection collection(s, _countof(s), data, _countof(data));\
        return collection;\
    }\
    virtual const ist::MemberInfoCollection& GetMemberInfo() { return _GetMemberInfo(); }

#define IST_MEMBER(name) \
    ist::CreateMemberInfo(&self_t::name, #name),

#define IST_SUPER(name) \
    ist::CreateSuperMembers< self_t, name >(name::_GetMemberInfo()),


    template<class T> inline void assign(T &a, const T &b) { a=b; }
    template<class T, size_t N> inline void assign(T (&a)[N], const T (&b)[N]) { std::copy(b, b+_countof(b), a); }

    template<class T> inline size_t get_size(const T &v) { return sizeof(v); }

    struct IMemberInfo
    {
        virtual ~IMemberInfo() {}
        virtual const char* GetName() const=0;
        virtual const void* GetValue(const void *obj) const=0;
        virtual size_t GetSize(const void *obj) const=0; // std::vector<> とかで動的にサイズ変わる可能性あるので一応 obj を引数に取るように
        virtual void SetValue(void *obj, const void *v)=0;
    };

    template<class T, class MemT>
    struct MemberInfo : public IMemberInfo
    {
        MemT T::*data;
        const char *name;
        MemberInfo(MemT T::*d, const char *n) : data(d), name(n) {}
        virtual const char* GetName() const                 { return name; }
        virtual const void* GetValue(const void *obj) const { return &(reinterpret_cast<const T*>(obj)->*data); }
        virtual size_t GetSize(const void *obj) const       { return get_size(reinterpret_cast<const T*>(obj)->*data); }
        virtual void SetValue(void *obj, const void *v)     { assign((reinterpret_cast<T*>(obj)->*data), *reinterpret_cast<const MemT*>(v)); }
    };

    template<class T, class MemT>
    IMemberInfo* CreateMemberInfo(MemT T::*data, const char *name)
    {
        return new MemberInfo<T,MemT>(data, name);
    }

    template<class Self, class Super>
    struct SuperMemberInfo : public IMemberInfo
    {
        IMemberInfo *mi;
        SuperMemberInfo(IMemberInfo *v) : mi(v) {}
        virtual const char* GetName() const                 { return mi->GetName(); }
        virtual const void* GetValue(const void *obj) const { return mi->GetValue( static_cast<const Super*>(reinterpret_cast<const Self*>(obj)) ); }
        virtual size_t GetSize(const void *obj) const       { return mi->GetSize( static_cast<const Super*>(reinterpret_cast<const Self*>(obj)) ); }
        virtual void SetValue(void *obj, const void *v)     { mi->SetValue( static_cast<Super*>(reinterpret_cast<Self*>(obj)), v); }
    };

    struct MemberInfoCollection
    {
        typedef std::vector<IMemberInfo*> container;
        typedef container::iterator iterator;
        typedef container::const_iterator const_iterator;
        container data;

        MemberInfoCollection() {}
        MemberInfoCollection(IMemberInfo **d, size_t n)
        {
            data.insert(data.end(), d, d+n);
        }
        MemberInfoCollection(const MemberInfoCollection *supers, size_t sn,  IMemberInfo **d, size_t n)
        {
            for(size_t i=0; i<sn; ++i) {
                data.insert(data.end(), supers[i].begin(), supers[i].end());
            }
            data.insert(data.end(), d, d+n);
        }

        IMemberInfo* operator[](int i) const { return data[i]; }
        size_t size() const { return data.size(); }
        iterator begin() { return data.begin(); }
        iterator end()   { return data.end(); }
        const_iterator begin() const { return data.begin(); }
        const_iterator end() const   { return data.end(); }
    };

    template<class Self, class Super>
    MemberInfoCollection CreateSuperMembers(const MemberInfoCollection& s)
    {
        MemberInfoCollection r;
        for(size_t i=0; i<s.data.size(); ++i) {
            r.data.push_back(new SuperMemberInfo<Self, Super>(s.data[i]));
        }
        return r;
    }

} // namespace ist
#endif // __ist_Base_Serialize__
