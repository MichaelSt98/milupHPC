#include "../../include/utils/timer.h"

//Timer::Timer() : m_beg(clock_t::now()) {
//
//}

Timer::Timer() {
    timer.restart();
}

void Timer::reset() {
    //m_beg = clock_t::now();
    timer.restart();
}

//double Timer::elapsed() const {
//    return std::chrono::duration_cast<millisecond_t>(clock_t::now() - m_beg).count();
//    //return std::chrono::duration_cast<second_t>(clock_t::now() - m_beg).count();
//    //return (clock_t::now() - m_beg).count();
//}

double Timer::elapsed() const {
    return timer.elapsed() * 1000;
}
