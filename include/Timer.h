#include <chrono> // for std::chrono functions

#ifndef CPP_TEMPLATE_PROJECT_TIMER_H
#define CPP_TEMPLATE_PROJECT_TIMER_H


class Timer {
private:
    // Type aliases to make accessing nested type easier
    using clock_t = std::chrono::high_resolution_clock;
    using second_t = std::chrono::duration<double, std::ratio<1> >;

    std::chrono::time_point<clock_t> m_beg;

public:

    Timer();
    void reset();
    double elapsed() const;

};


#endif //CPP_TEMPLATE_PROJECT_TIMER_H
