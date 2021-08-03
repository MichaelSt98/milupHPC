//
// Created by Michael Staneker on 18.12.20.
//

#include "../include/Logger.h"

Color::Modifier::Modifier(Code pCode) : code(pCode) {}

std::ostream& Color::operator<<(std::ostream& os, const Color::Modifier& mod) {
    return os << "\033[" << mod.code << "m";
}

Logger::Logger(typelog type) {
    msglevel = type;
    Color::Modifier def(Color::FG_DEFAULT);
    if(LOGCFG.headers) {
        std::cout << getColor(type);
        operator << (getLabel(type));
        std::cout << def;
    }
}

Logger::~Logger() {
    if(opened) {
        std::cout << std::endl;
    }
    opened = false;
}

inline std::string Logger::getLabel(typelog type) {
    std::string label;
    std::string rankLbl = "";
    if (LOGCFG.myrank >= 0){
        rankLbl = "(" + std::to_string(LOGCFG.myrank) + ")";
    }
    switch(type) {
        case DEBUG: label = "[DEBUG] "; break;
        case INFO:  label = "[INFO ] "; break;
        case WARN:  label = "[WARN ] "; break;
        case ERROR: label = "[ERROR] "; break;
    }
    return rankLbl + label;
}

inline Color::Modifier Logger::getColor(typelog type) {
    Color::Modifier color(Color::FG_DEFAULT);
    switch(type) {
        case DEBUG: color.code = Color::FG_DARK_GRAY; break;
        case INFO:  color.code = Color::FG_LIGHT_GREEN; break;
        case WARN:  color.code = Color::FG_YELLOW; break;
        case ERROR: color.code = Color::FG_RED; break;
    }
    return color;
}
