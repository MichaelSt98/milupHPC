#ifndef VECTOR3_H
#define VECTOR3_H

#include <iostream>
#include <iomanip>
#include <cmath>
#include <boost/mpi.hpp>

template <class T>
class Vector3 {

public:

    T x;
    T y;
    T z;

    // standard constructor with default values
    Vector3() : x(T(0)), y(T(0)), z(T(0)) {}

    // standard constructor
    Vector3(const T _x, const T _y, const T _z) : x(_x), y(_y), z(_z) {}

    // standard constructor with same value for all entries
    Vector3(const T s) : x(s), y(s), z(s) {}

    // copy constructor
    Vector3(const Vector3& src) : x(src.x), y(src.y), z(src.z) {}

    // move constructor
    Vector3(Vector3&& src) : x(src.x ), y(src.y), z(src.z) {
        //src = nullptr;
    }

    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & x;
        ar & y;
        ar & z;
    }

    typedef T DataType;

    const Vector3& operator=(const Vector3& rhs) {
        x = rhs.x;
        y = rhs.y;
        z = rhs.z;
        return (*this);
    }

    const Vector3 & operator=(const T s) {
        x = y = z = s;
        return (*this);
    }

    Vector3 operator+(const Vector3& rhs) const {
        return Vector3(x + rhs.x, y + rhs.y, z + rhs.z);
    }

    const Vector3& operator+=(const Vector3& rhs) {
        (*this) = (*this) + rhs;
        return (*this);
    }

    Vector3 operator-(const Vector3& rhs) const {
        return Vector3(x - rhs.x, y - rhs.y, z - rhs.z);
    }

    const Vector3 & operator-=(const Vector3 & rhs) {
        (*this) = (*this) - rhs;
        return (*this);
    }

    Vector3 operator*(const T s) const {
        return Vector3(x * s, y * s, z * s);
    }

    const Vector3& operator*=(const T s) {
        (*this) = (*this) * s;
        return (*this);
    }

    friend Vector3 operator*(const T s, const Vector3& v) {
        return (v * s);
    }

    Vector3 operator/(const T s) const {
        return Vector3(x/s, y/s, z/s);
    }

    const Vector3& operator/= (const T s) {
        (*this) = (*this) / s;
        return (*this);
    }

    const Vector3& operator+() const {
        return (* this);
    }

    const Vector3 operator-() const {
        return Vector3(-x, -y, -z);
    }

    T operator*(const Vector3& rhs) const {
        return (x * rhs.x) + (y * rhs.y) + (z * rhs.z);
    }

    Vector3 operator^(const Vector3& rhs) const {
        return Vector3( (y * rhs.z - z * rhs.y),
                        (z * rhs.x - x * rhs.z),
                        (x * rhs.y - y * rhs.x));
    }

    bool operator<(const Vector3& rhs) const {
        return (x < rhs.x && y < rhs.y && z < rhs.z);
    }

    bool operator<=(const Vector3& rhs) const {
        return (x <= rhs.x && y <= rhs.y && z <= rhs.z);
    }

    bool operator>(const Vector3& rhs) const {
        return (x > rhs.x && y > rhs.y && z > rhs.z);
    }

    bool operator>=(const Vector3& rhs) const {
        return (x >= rhs.x && y >= rhs.y && z >= rhs.z);
    }

    bool withinRadius(const Vector3& rhs, const T r) const {
        if (rhs.x < x+r && rhs.x > x-r &&
            rhs.y < y+r && rhs.y > y-r &&
            rhs.z < z+r && rhs.z > z-r) {
            return true;
        }
        return false;
    }

    Vector3 absolute() {
        return Vector3<T> {abs(x), abs(y), abs(z)};
    }


    template <typename U>
    operator Vector3<U> () const {
        return Vector3<U> (static_cast<U>(x),
                           static_cast<U>(y),
                           static_cast<U>(z));
    }

    T getMax() const {
        T max_val = (x > y) ? x : y;
        max_val = (max_val > z ) ? max_val : z;
        return max_val;
    }

    T getMin() const {
        T min_val = (x < y) ? x : y;
        min_val = (min_val < z ) ? min_val : z;
        return min_val;
    }

    template <class F>
    Vector3 applyEach(F f) const {
        return Vector3(f(x), f(y), f(z));
    }

    template <class F>
    friend Vector3 ApplyEach(F f, const Vector3& arg1, const Vector3& arg2) {
        return Vector3(f(arg1.x, arg2.x), f(arg1.y, arg2.y), f(arg1.z, arg2.z));
    }

    friend std::ostream& operator<<(std::ostream& out, const Vector3& u) {
        out << "(" << u.x << ", " << u.y << ", " << u.z << ")";
        return out;
    }

    friend std::istream & operator >>(std::istream& in, Vector3 & u){
        in >> u.x;
        in >> u.y;
        in >> u.z;
        return in;
    }

    T & operator[](const int i) {
        if (i == 0) {
            return x;
        } else if (i == 1) {
            return y;
        }
        if (i == 2) {
            return z;
        } else {
            std::cerr << "Vector element:" << i << " out of range!" << std::endl;
        }
    }

    T getMagnitude() const {
        return sqrt(x*x + y*y + z*z);
    }

    T getDistanceSquared(const Vector3& u) const {
        T dx = x - u.x;
        T dy = y - u.y;
        T dz = z - u.z;
        return dx*dx + dy*dy + dz*dz;
    }

    T getDistance(const Vector3& u) const {
        T getDistanceSquared(u);
        return 0;
        //return sqrt(T); //TODO: implement sqrt
    }

    bool operator==(const Vector3& u) const {
        return ((x==u.x) && (y==u.y) && (z==u.z));
    }

    bool operator!=(const Vector3& u) const {
        return ((x!=u.x) || (y!=u.y) || (z!=u.z));
    }

    //friend std::ostream &operator << (std::ostream &os, const Vector3<T> &v);

};

template <>
inline Vector3<float> Vector3<float>::operator/(const float s) const {
    const float inv_s = 1.0f/s;
    return Vector3(x * inv_s, y * inv_s, z * inv_s);
}
template <>
inline Vector3<double> Vector3<double>::operator/(const double s) const {
    const double inv_s = 1.0/s;
    return Vector3(x * inv_s, y * inv_s, z * inv_s);
}

/*inline std::ostream &operator<<(std::ostream &os, const Vector3 &v)
{
    os << '(' << v.x << ',' << v.y << ',' << v.z << ')';
    return os;
}*/


#endif //VECTOR3_H
