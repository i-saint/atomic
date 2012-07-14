#ifndef __ist_Base_Serialize__
#define __ist_Base_Serialize__

#include <cstdlib>
#include <vector>
#include <algorithm>

#define IST_INTROSPECTION(...) \
    static const ist::MemberInfoCollection& _GetMemberInfo() { \
        static bool initialized = false;\
        static ist::MemberInfoCollection collection;\
        if(!initialized) {\
            initialized = true;\
            __VA_ARGS__\
        }\
        return collection;\
    }\
    virtual const ist::MemberInfoCollection& GetMemberInfo() { return _GetMemberInfo(); }

#define IST_NAME(name) \
    collection.setName(#name);

#define IST_SUPER(name) \
    collection.addSuper(ist::CreateSuperMembers<this_t, name>(name::_GetMemberInfo()));

#define IST_MEMBER(name) \
    collection.addMember(ist::CreateMemberInfo(&this_t::name, #name));


namespace ist {

    template<class T>
    inline void assign(T &a, const T &b)
    {
        a = b;
    }

    template<class T, size_t N>
    inline void assign(T (&a)[N], const T (&b)[N])
    {
        for(size_t i=0; i<N; ++i) {
            assign<T>(a[i], b[i]);
        }
    }

    template<class T>
    inline size_t get_size(const T &v)
    {
        return sizeof(v);
    }




    struct IMemberInfo
    {
        virtual ~IMemberInfo() {}
        virtual const char* GetName() const=0;
        virtual const void* GetValue(const void *obj) const=0;
        virtual size_t GetSize(const void *obj) const=0; // POD 型は sizeof するだけなので静的に決まるが、std::vector<> とかは動的にサイズ変わるので obj を引数に取る必要がある
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
        const char *name;

        MemberInfoCollection() : name("") {}
        void setName(const char *v) { name=v; }
        void addSuper(const MemberInfoCollection &supers)
        {
            data.insert(data.end(), supers.begin(), supers.end());
        }
        void addMember(IMemberInfo *d)
        {
            data.push_back(d);
        }

        const char* getName() { return name; }
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
