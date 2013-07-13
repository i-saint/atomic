#include "stdafx.h"
#include "../FunctionID.h"
#include "Game/AtomicApplication.h"
#include "Game/EntityClass.h"
#include "WebServer.h"
#include "LevelEditorCommandHandler.h"

namespace atm {

#ifdef atm_enable_LevelEditorServer

struct NucleiCommandHandler_Initializer
{
    NucleiCommandHandler_Initializer() {
        NucleiCommandHandler::getHandlerTable();
    }
} g_NucleiCommandHandler_Initializer;

NucleiCommandHandler::HandlerTable& NucleiCommandHandler::getHandlerTable()
{
    static HandlerTable s_table;
    if(s_table.empty()) {
        s_table["/nuclei/call"]   = [](NucleiCommandHandler *o, HTTPServerRequest &req, HTTPServerResponse &res){ o->handleCall(req,res);   };
        s_table["/nuclei/create"] = [](NucleiCommandHandler *o, HTTPServerRequest &req, HTTPServerResponse &res){ o->handleCreate(req,res); };
        s_table["/nuclei/delete"] = [](NucleiCommandHandler *o, HTTPServerRequest &req, HTTPServerResponse &res){ o->handleDelete(req,res); };
        s_table["/nuclei/state"]  = [](NucleiCommandHandler *o, HTTPServerRequest &req, HTTPServerResponse &res){ o->handleState(req,res);  };
        s_table["/nuclei/entity"] = [](NucleiCommandHandler *o, HTTPServerRequest &req, HTTPServerResponse &res){ o->handleEntity(req,res); };
        s_table["/nuclei/const"]  = [](NucleiCommandHandler *o, HTTPServerRequest &req, HTTPServerResponse &res){ o->handleConst(req,res);  };
    }
    return s_table;
}

NucleiCommandHandler::Handler* NucleiCommandHandler::findHandler(const stl::string &path)
{
    auto &table = getHandlerTable();
    auto i = table.find(path);
    return i==table.end() ? nullptr : &i->second;
}

FunctionID NucleiCommandHandler::getValidFID(const stl::string &name)
{
    if(name.size()<64) {
        if(atmIsEditMode()) {
            char fid[128];
            istSPrintf(fid, "FID_%s", name.c_str());
            return GetFunctionIDNum(fid);
        }
        else {
            if(name=="instruct") { return FID_instruct; }
        }
    }
    return FID_unknown;
}

bool NucleiCommandHandler::isDeployable(EntityClassID ecid)
{
    if(EntityClassInfo *eci=GetEntityClassInfo(ecid)) {
        if(atmIsEditMode()) { return true; }
        else if(eci->deploy==DF_RTS) { return true; }
    }
    return false;
}


NucleiCommandHandler::NucleiCommandHandler()
{
}

void NucleiCommandHandler::handleRequest(HTTPServerRequest &request, HTTPServerResponse &response)
{
    if(Handler *hanlder = findHandler(request.getURI())) {
        (*hanlder)(this, request, response);
    }
}

void NucleiCommandHandler::respondCode(HTTPServerResponse &response, int32 code)
{
    char str[16];
    size_t len;
    istSPrintf(str, "%d", code);
    len = strlen(str);

    response.setContentType("text/plain");
    response.setContentLength(len);
    std::ostream &ostr = response.send();
    ostr.write(str, len);
}

void NucleiCommandHandler::handleCreate(HTTPServerRequest &request, HTTPServerResponse &response)
{
    std::string data;
    GetDecodedRequestBody(request, data);
    int32 code = 0;

    std::smatch m1;
    if(std::regex_search(data, m1, std::regex("classid=(\\d+),pos=(.+)"))) {
        EntityClassID ecid = (EntityClassID)_atoi64(m1[1].str().c_str());
        if(!isDeployable(ecid)) {
            code=RC_InvalidCommand; goto RESPOND;
        }

        variant vpos;
        if(ParseArg(vpos, m1[2].str())) {
            vec2 pos = (vec2&)vpos + WebServer::getInstance()->randomVec2()*0.01f;
            {
                LevelEditorCommand_Create cmd;
                cmd.classid = ecid;
                WebServer::getInstance()->pushCommand((LevelEditorCommand&)cmd);
            }
            {
                LevelEditorCommand_Call cmd;
                cmd.entity = 0;
                cmd.function = FID_move;
                cmd.arg = vec3(pos, 0.0f);
                WebServer::getInstance()->pushCommand((LevelEditorCommand&)cmd);
            }
        }
    }
RESPOND:
    respondCode(response, code);
}

void NucleiCommandHandler::handleDelete(HTTPServerRequest &request, HTTPServerResponse &response)
{
    std::string data;
    GetDecodedRequestBody(request, data);

    uint32 h = 0;
    sscanf(data.c_str(), "entity=%u", &h);

    LevelEditorCommand_Delete cmd;
    cmd.type = LEC_Delete;
    cmd.entity =  h;
    WebServer::getInstance()->pushCommand((LevelEditorCommand&)cmd);

    respondCode(response, 0);
}

void NucleiCommandHandler::handleCall(HTTPServerRequest &request, HTTPServerResponse &response)
{
    std::string data;
    GetDecodedRequestBody(request, data);
    int32 code = 0;

    std::smatch m1;
    if(std::regex_search(data, m1, std::regex("entity=(\\d+),func=(\\w+),arg=(.+)"))) {
        variant arg;
        EntityHandle entity = (EntityHandle)_atoi64(m1[1].str().c_str());
        FunctionID fid = getValidFID(m1[2].str());
        if(fid!=0 && ParseArg(arg, m1[3].str())) {
            LevelEditorCommand_Call cmd;
            cmd.entity = entity;
            cmd.function = fid;
            cmd.arg = arg;
            WebServer::getInstance()->pushCommand((LevelEditorCommand&)cmd);
        }
        else {
            code = RC_InvalidCommand;
        }
    }
    respondCode(response, code);
}

void NucleiCommandHandler::handleState(HTTPServerRequest &request, HTTPServerResponse &response)
{
    LevelEditorQuery q;
    q.type = LEQ_Entities;
    WebServer::getInstance()->pushQuery(q);
    while(!q.completed && !WebServer::getInstance()->endFlag()) {
        ist::MiliSleep(5);
    }

    response.setContentType("application/octet-stream");
    response.setContentLength(q.response.size());
    std::ostream &ostr = response.send();
    ostr << q.response;
}


void NucleiCommandHandler::handleEntity(HTTPServerRequest &request, HTTPServerResponse &response)
{
    std::string data;
    GetDecodedRequestBody(request, data);

    uint32 h = 0;
    sscanf(data.c_str(), "entity=%u", &h);

    LevelEditorQuery q;
    q.type = LEQ_Entity;
    q.optional =  h;
    WebServer::getInstance()->pushQuery(q);
    while(!q.completed && !WebServer::getInstance()->endFlag()) {
        ist::MiliSleep(5);
    }

    response.setContentType("application/json");
    response.setContentLength(q.response.size());
    std::ostream &ostr = response.send();
    ostr << q.response;
}

void NucleiCommandHandler::handleConst(HTTPServerRequest &request, HTTPServerResponse &response)
{
    stl::string ret;
    ret += "{";
    {
        bool first = true;
        ret += "\"entityTypes\":{";
        EntityClassIDEachPair([&](const ist::EnumStr &es){
            EntityClassID ecid = (EntityClassID)es.num;
            const EntityClassInfo *eci = GetEntityClassInfo(ecid);
            bool deployable = false;
            float32 cost = 0.0f;
            if(eci) {
                if(eci->deploy==DF_RTS || (eci->deploy==DF_Editor && atmIsEditMode())) { deployable=true; }
                cost = eci->cost;
            }
            if(!first){ret+=",";} first=false;
            ret += ist::Format(
                "\"%s\":{\"id\":%d,\"deployable\":%d,\"cost\":%f}",
                es.str+3, es.num, deployable, cost );

        });
        ret += "},";
    }
    {
        bool first = true;
        ret += "\"functions\":{";
        FunctionIDEachPair([&](const ist::EnumStr &es){
            if(!first){ret+=",";} first=false;
            ret += ist::Format("\"%s\":%d", es.str+4, es.num);
        });
        ret += "}";
    }
    ret += "}";

    response.setContentType("application/json");
    response.setContentLength(ret.size());
    std::ostream &ostr = response.send();
    ostr << ret;
}

#else // atm_enable_LevelEditorServer
#endif // atm_enable_LevelEditorServer
} // namespace atm
