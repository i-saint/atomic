#include "interface.h"

class Hoge : public IHoge
{
private:
    int m_value;

    friend class boost::serialization::access;
    template<class A>
    void serialize(A &ar, const uint32_t version)
    {
        ar & boost::serialization::base_object<IHoge>(*this);
        ar & m_value;
    }

public:
    Hoge() : m_value(100)
    {
    }

    void doSomething()
    {
        printf("Hoge::doSomething(): %d\n", m_value);
        ++m_value;
    }
};
BOOST_CLASS_EXPORT(Hoge)

InterModule IHoge* CreateHoge() { return new Hoge(); }
