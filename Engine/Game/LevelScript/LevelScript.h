#ifndef atm_Engine_Game_LevelScript_h
#define atm_Engine_Game_LevelScript_h

namespace atm {


class LevelScript
{
public:
    enum EventType {
        Evt_EntityCreate,
        Evt_EntityDelete,
        Evt_EntityCall,
        Evt_EntityCallWithSlot,
    };
    struct EventBase
    {
        EventType type;
        char data[44];
    };

    struct EventCreate
    {
        EventType type;
        uint32 entity_class;
        uint32 entity_slot;
    };
    struct EventDelete
    {
        EventType type;
        uint32 entity_slot;
    };
    struct EventCall
    {
        EventType type;
        uint32 entity_slot;
        uint32 function_id;
        variant32 arg;
    };
    struct EventCallWithSlot
    {
        EventType type;
        uint32 entity_slot;
        uint32 function_id;
        uint32 arg_entity_slot;
    };


private:

};

} // namespace atm
#endif // atm_Engine_Game_LevelScript_h
