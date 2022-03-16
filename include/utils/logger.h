/**
 * @file logger.h
 * @brief C++ style logger
 *
 *
 * @author Michael Staneker
 * @bug no known bugs
 */
#ifndef NBODY_LOGGER_H
#define NBODY_LOGGER_H

#include <iostream>
#include <string>
#include "color.h"
#include "../parameter.h"
#include <iostream>
#include <fstream>

namespace Color {
    /**
     * @brief Modify (color) of terminal output.
     */
    class Modifier {
    public:
        Code code;
        /**
         * @brief Constructor
         *
         * @param pCode color code
         */
        Modifier(Code pCode);
        /**
         * @brief Ofstream operator overload.
         *
         * @param os
         * @param mod
         * @return
         */
        friend std::ostream& operator<<(std::ostream& os, const Color::Modifier& mod);
    };
}

/// logging types
enum typeLog {
    DEBUG, /// debug log type
    INFO, /// info log type
    TRACE, /// trace log type
    WARN, /// warning log type
    ERROR, /// error log type
    TIME /// time log type
};

/**
 * @brief Logger settings.
 */
struct structLog {
    /// show headers
    bool headers = false;
    /// Minimum logging level to be shown
    typeLog level = DEBUG;
    /// whether to use MPI
    int rank = -1; // don't use MPI by default
    /// MPI rank to be displayed (default: -1 -> display all)
    int outputRank = -1;
    /// write additionally to log file
    bool write2LogFile = true;
    /// log file to be written
    std::string logFileName {"log/miluphpc.log"};
    /// omit time output/logging
    bool omitTime = false;
};

extern structLog LOGCFG;

/**
 * @brief Logger class.
 */
class Logger {
public:
    /**
     * Default constructor.
     */
    Logger() {}
    /**
     * @brief Constructor.
     *
     * @param type logging type
     * @param toLog
     */
    Logger(typeLog type, bool toLog=false);
    ~Logger();

    /**
     * @brief Log/output any message.
     *
     * @tparam T message data type(s)
     * @param msg message to be logged
     * @return
     */
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

    /**
     * @brief Specialized log/output message for keyType (unsigned long)
     * @param key
     * @return
     */
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
    typeLog msgLevel = INFO;
    inline std::string getLabel(typeLog type);
    inline Color::Modifier getColor(typeLog type);
    std::ofstream logFile;
    bool openedLogFile = false;
    bool toLog = false;
    bool omit = false;
};

#endif //NBODY_LOGGER_H
