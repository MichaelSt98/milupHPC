#ifndef NBODY_LOGGER_H
#define NBODY_LOGGER_H

#include <iostream>
#include <string>
#include "color.h"
#include "../parameter.h"
#include <iostream>
#include <fstream>

namespace Color {
    class Modifier {
    public:
        Code code;
        Modifier(Code pCode);
        friend std::ostream& operator<<(std::ostream& os, const Color::Modifier& mod);
    };
}

enum typeLog {
    DEBUG,
    WARN,
    ERROR,
    INFO,
    TIME
};

struct structLog {
    bool headers = false;
    typeLog level = DEBUG;
    int rank = -1; // don't use MPI by default
    int outputRank = -1;
    bool write2LogFile = true;
    std::string logFileName {"log/miluphpc.log"};
    bool omitTime = false;
};

extern structLog LOGCFG;

class Logger {
public:
    Logger() {}
    Logger(typeLog type, bool toLog=false);
    ~Logger();

    template<class T> Logger &operator<<(const T &msg) {

        if (LOGCFG.write2LogFile && (this->toLog || (msgLevel == typeLog::WARN || msgLevel == typeLog::ERROR))) {
            logFile << msg;
            openedLogFile = true;
        }
        if (msgLevel >= LOGCFG.level && (LOGCFG.rank == LOGCFG.outputRank || LOGCFG.outputRank == -1)) {
            if (!omit) {
                std::cout << msg;
                opened = true;
            }
        }
        return *this;
    }

    Logger &operator<<(const unsigned long &key) {
        int level = 21;
        if (msgLevel >= LOGCFG.level && (LOGCFG.rank == LOGCFG.outputRank || LOGCFG.outputRank == -1)) {
            int levels [level];
            for (int i = 0; i<level; i++) {
                levels[i] = (key >> DIM*i) & (unsigned long)(POW_DIM - 1);
            }
            std::string msg = "#|";
            for (int i = level-1; i>=0; i--) {
                msg += std::to_string(levels[i]);
                msg += "|";
            }
            if (!omit) {
                std::cout << msg;
                opened = true;
            }
        }
        return *this;
    }


private:
    bool opened = false;
    typeLog msgLevel = DEBUG;
    inline std::string getLabel(typeLog type);
    inline Color::Modifier getColor(typeLog type);
    std::ofstream logFile;
    bool openedLogFile = false;
    bool toLog = false;
    bool omit = false;
};

#endif //NBODY_LOGGER_H
