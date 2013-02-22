#ifndef atomic_Game_Entity_Routine_h
#define atomic_Game_Entity_Routine_h
namespace atomic {

enum RoutineClassID
{
    RCID_Null,

    RCID_Routine_Shoot,
    RCID_Routine_HomingPlayer,
    RCID_Routine_Pinball,

    RCID_End,
};

class IRoutine;
typedef IRoutine* (*RoutineCreator)();
typedef RoutineCreator (RoutineCreatorTable)[RCID_End];

RoutineCreatorTable& GetRoutineCreatorTable();
IRoutine* CreateRoutine(RoutineClassID rcid);

template<class RoutineType> IRoutine* CreateRoutine();
template<class RoutineType> class AddRoutineTable;

#define atomicImplementRoutine(Class) \
    template<> IRoutine* CreateRoutine<Class>() { return istNew(Class)(); } \
    template<> struct AddRoutineTable<Class> {\
        AddRoutineTable() { GetRoutineCreatorTable()[RCID_##Class] = &CreateRoutine<Class>; }\
    };\
    AddRoutineTable<Class> g_add_routine_creator_##Class;


class IEntity;

class IRoutine
{
protected:
    IEntity *m_obj;

    istSerializeBlock(
        istSerialize(m_obj)
        )

public:
    IRoutine()  : m_obj(NULL) {}
    virtual ~IRoutine() {}
    IEntity* getEntity() { return m_obj; }
    void setEntity(IEntity *v) { m_obj=v; }

    virtual void update(float32 dt)     {}
    virtual void asyncupdate(float32 dt){}
    virtual void draw() {}

    virtual bool call(FunctionID call_id, const variant &v) { return false; }
    virtual bool query(FunctionID query_id, variant &v) const { return false; }
};




} // namespace atomic
#endif // atomic_Game_Entity_Routine_h
