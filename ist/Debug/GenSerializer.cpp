#include "istPCH.h"
#include "ist/Base/StringUtil.h"
#include "GenSerializer.h"

namespace ist {

_GenSerializerProcess::instance_cont& _GenSerializerProcess::getInstances()
{
    static instance_cont s_instances;
    return s_instances;
}

_GenSerializerProcess::_GenSerializerProcess(const char *path, const callback_t &cb)
{
    getInstances().push_back(this);
    m_path = path;
    m_callback = cb;
}

void _GenSerializerProcess::process()
{
    std::string cpp;
    if(FILE *fin=fopen(m_path, "rb")) {
        if(!fin) { return; }
        fseek(fin, 0, SEEK_END);
        cpp.resize((size_t)ftell(fin));
        fseek(fin, 0, SEEK_SET);
        fread(&cpp[0], 1, cpp.size(), fin);
        fclose(fin);
    }

    cpp = gsub(cpp, std::regex("^[ \\t]*GenSerializerBegin\\((.+?)\\)[.\\r\\n]+?GenSerializerEnd\\(\\)"), [&](const std::cmatch &m)->std::string {
        std::string class_name = m[1].str();
        std::string code;
        EachMembersByTypeName(class_name.c_str(), [&](const MemberInfo *mi, size_t num){
            code = m_callback(class_name.c_str(), mi, num);
        });

        std::string rep;
        rep += "GenSerializerBegin(";
        rep += class_name;
        rep += ")\r\n";
        rep += code;
        rep += "GenSerializerEnd()";
        return rep;
    });
    if(FILE *fout=fopen(m_path, "wb")) {
        fwrite(cpp.c_str(), cpp.size(), 1, fout);
        fclose(fout);
    }
}

void _GenSerializerProcess::exec()
{
    instance_cont& instances = getInstances();
    for(size_t i=0; i<instances.size(); ++i) {
        instances[i]->process();
    }
}


} // namespace ist
