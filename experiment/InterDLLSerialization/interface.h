#pragma warning(push)
#pragma warning(disable: 4244)
#   define BOOST_SERIALIZATION_DYN_LINK
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/export.hpp>
#pragma warning(pop)
#include <fstream>

#ifdef BuildDLL
#   define InterModule __declspec(dllexport)
#else // BuildDLL
#   define InterModule __declspec(dllimport)
#endif // BuildDLL


class IHoge
{
private:
    friend class boost::serialization::access;
    template<class A>
    void serialize(A &ar, const uint32_t version)
    {
    }

public:
    virtual ~IHoge() {}
    virtual void release() { delete this; }
    virtual void doSomething()=0;
};

InterModule IHoge* CreateHoge();
