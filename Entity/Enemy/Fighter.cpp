#include "entityPCH.h"
#include "Entity/EntityCommon.h"
#include "Entity/Enemy.h"
#include "Entity/Routine.h"

namespace atm {

class BreakableParts : public Breakable<Entity_Direction>
{
typedef Breakable<Entity_Direction> super;
private:
    istSerializeBlock(
        istSerializeBase(super)
    )

public:
};
atmImplementEntity(BreakableParts);
atmExportClass(BreakableParts);


class BreakableCore : public Breakable<Entity_Direction>
{
typedef Breakable<Entity_Direction> super;
private:
    istSerializeBlock(
        istSerializeBase(super)
    )

public:
};
atmImplementEntity(BreakableCore);
atmExportClass(BreakableCore);


class SmallFighter : public Breakable<Entity_Direction>
{
typedef Breakable<Entity_Direction> super;
private:
    istSerializeBlock(
        istSerializeBase(super)
    )

public:
};
atmImplementEntity(SmallFighter);
atmExportClass(SmallFighter);


class MediumFighter : public Breakable<Entity_Direction>
{
typedef Breakable<Entity_Direction> super;
private:
    istSerializeBlock(
        istSerializeBase(super)
    )

public:
};
atmImplementEntity(MediumFighter);
atmExportClass(MediumFighter);


class LargeFighter : public Breakable<Entity_Direction>
{
typedef Breakable<Entity_Direction> super;
private:
    istSerializeBlock(
        istSerializeBase(super)
    )

public:
};
atmImplementEntity(LargeFighter);
atmExportClass(LargeFighter);


class Shell : public Breakable<Entity_Direction>
{
typedef Breakable<Entity_Direction> super;
private:
    istSerializeBlock(
        istSerializeBase(super)
    )

public:
};
atmImplementEntity(Shell);
atmExportClass(Shell);


class Zab : public Breakable<Entity_Direction>
{
typedef Breakable<Entity_Direction> super;
private:
    istSerializeBlock(
        istSerializeBase(super)
    )

public:
};
atmImplementEntity(Zab);
atmExportClass(Zab);


class SmallNucleus : public Breakable<Entity_Direction>
{
typedef Breakable<Entity_Direction> super;
private:
    istSerializeBlock(
        istSerializeBase(super)
    )

public:
};
atmImplementEntity(SmallNucleus);
atmExportClass(SmallNucleus);



} // namespace atm
