#include "../../include/utils/logger.h"

Color::Modifier::Modifier(Code pCode) : code(pCode) {}

std::ostream& Color::operator<<(std::ostream& os, const Color::Modifier& mod) {
    return os << "\033[" << mod.code << "m";
}

Logger::Logger(typeLog type, bool toLog) {
    if (LOGCFG.omitTime && type == typeLog::TIME) {
        omit = true;
    }
    this->toLog = toLog;
    msgLevel = type;
    if (LOGCFG.write2LogFile && (this->toLog || (msgLevel == typeLog::WARN || msgLevel == typeLog::ERROR))) {
        logFile.open(LOGCFG.logFileName.c_str(), std::ios::out | std::ios::app);

    }
    Color::Modifier def(Color::FG_DEFAULT);
    if(LOGCFG.headers) {
        if (!omit) {
            std::cout << getColor(type);
            operator<<(getLabel(type));
            std::cout << def;
        }
    }
}

Logger::~Logger() {
    if (opened && !omit) {
        std::cout << std::endl;
    }
    if (openedLogFile) {
        logFile << std::endl;
        logFile.close();
    }
    opened = false;
    openedLogFile = false;
}

inline std::string Logger::getLabel(typeLog type) {
    std::string label;
    std::string rankLbl = "";
    if (LOGCFG.rank >= 0){
        rankLbl = "(" + std::to_string(LOGCFG.rank) + ")";
    }
    switch(type) {
        case DEBUG: label = "[DEBUG] "; break;
        case INFO:  label = "[INFO ] "; break;
        case WARN:  label = "[WARN ] "; break;
        case ERROR: label = "[ERROR] "; break;
        case TIME:  label = "[TIME ] "; break;
    }
    return rankLbl + label;
}

inline Color::Modifier Logger::getColor(typeLog type) {
    Color::Modifier color(Color::FG_DEFAULT);
    switch(type) {
        case DEBUG: color.code = Color::FG_DARK_GRAY; break;
        case INFO:  color.code = Color::FG_LIGHT_GREEN; break;
        case WARN:  color.code = Color::FG_YELLOW; break;
        case ERROR: color.code = Color::FG_RED; break;
        case TIME:  color.code = Color::FG_BLUE; break;
    }
    return color;
}
