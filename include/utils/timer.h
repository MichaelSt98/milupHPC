/**
 * @file timer.h
 * @brief C++ timer based on boost::mpi::timer.
 *
 *
 * @author Michael Staneker
 * @bug no known bugs
 */
#ifndef MILUPHPC_TIMER_H
#define MILUPHPC_TIMER_H

#include <chrono>
#include <boost/mpi.hpp>

class Timer {

private:

    using clock_t = std::chrono::high_resolution_clock;
    using millisecond_t = std::chrono::milliseconds;
    using second_t = std::chrono::duration<double, std::ratio<1> >;

    //std::chrono::time_point<clock_t> m_beg;
    boost::mpi::timer timer;

public:

    /**
     * @brief Constructor.
     */
    Timer();
    /**
     * @brief Reset timer instance.
     */
    void reset();
    /**
     * @brief Get elapsed time since instantiation/latest reset.
     *
     * @return elapsed time since instantiation/latest reset
     */
    double elapsed() const;

};


#endif //MILUPHPC_TIMER_H
