#include "interface.h"

BOOST_CLASS_EXPORT(IHoge)

void serialize()
{
    printf("serialize()\n");
    IHoge *hoge = CreateHoge();
    hoge->doSomething();

    std::ofstream fs("hoge.bin", std::ios::binary);
    boost::archive::binary_oarchive ar(fs);
    ar << hoge;
}

void deserialzie()
{
    printf("deserialzie()\n");

    IHoge *hoge = nullptr;

    std::ifstream fs("hoge.bin", std::ios::binary);
    boost::archive::binary_iarchive ar(fs);
    ar >> hoge;

    hoge->doSomething();
}

int main(int argc, char *argv[])
{
    serialize();
    deserialzie();
}
