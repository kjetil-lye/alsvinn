#pragma once
#include <vector>
#include "alsfvm/vec1.hpp"
namespace alsfvm {

    ///
    /// Small vector class to hold 3D data
    ///
    template<class T>
    struct vec3 {
        T x;
        T y;
        T z;

        __device__ __host__ vec3()
            : x(0), y(0), z(0)
        {

        }

        __device__ __host__ vec3(T t) :
            x(t), y(t), z(t) 
        {

        }
        __device__ __host__ vec3(T x, T y, T z)
        : x(x), y(y), z(z)
        {
            // Empty
        }

        __device__ __host__ vec3(T x, vec1<T> y, T z)
            : x(x), y(y.x), z(z)
        {
            // Empty
        }

        __device__ __host__ vec3(const vec3<T&>& other)
            : x(other.x), y(other.y), z(other.z)
        {

        }

        template<class S>
        __device__ __host__ vec3& operator=(const vec3<S>& other) {
            x = other.x;
            y = other.y;
            z = other.z;

            return *this;
        }

        __device__ __host__ bool operator==(const vec3& other) const {
            return other.x == x && other.y == y && other.z == z;
        }
#if __cplusplus > 199711L || WIN32
        ///
        /// Converts the vector to an std::vector<T>
        /// output is {x, y, z}
        ///
        std::vector<T> toStdVector() {
            return std::vector<T>({ x, y, z });
        }
#endif
        ///
        /// Converts the vector to the other type
        ///
        template<class S>
        __device__ __host__ vec3<S> convert() const  {
            return vec3<S>(S(x), S(y), S(z));
        }

        ///
        /// Returns the i-th component of the vector.
        ///
        __device__ __host__ const T& operator[](size_t i) const {
            // Note: We only store three numbers in this struct, hence this is safe
            return ((T*)this)[i];
        }

        ///
        /// Returns the i-th component of the vector.
        ///
        __device__ __host__ T& operator[](size_t i) {
            // Note: We only store three numbers in this struct, hence this is safe
            return ((T*)this)[i];
        }

        ///
        /// Computes the dot (scalar) product
        ///
        __device__ __host__ T dot(const vec3<T>& other) const {
            return x*other.x + y*other.y + z*other.z;
        }

        ///
        /// Returns 3 (number of components)
        ///
        __device__ __host__ static constexpr size_t size()  {
            return 3;
        }

        template<class S>
        __device__ __host__ inline vec3<T>& operator+=( const vec3<S>& b) {
            x+= b.x;
            y+= b.y;
            z+=b.z;

            return *this;
        }
    };

    ///
    /// Computes the component wise division of a by b.
    /// Ie. the new vector will be
    /// \f[(a_0/b_0, a_1/b_1, a_2/b_2)\f]
    ///
    template<class T>
    __device__ __host__ inline vec3<T> operator/(const vec3<T>& a, const vec3<T>& b) {
        return vec3<T>(a.x / b.x, a.y / b.y, a.z / b.z);
    }

    ///
    /// Computes the product \f$\mathrm{scalar} \vec{a}\f$
    /// \note Creates a new vector instance
    ///
    template<class T>
    __device__ __host__ inline vec3<T> operator*(T scalar, const vec3<T>& a) {
        return vec3<T>(a.x*scalar, a.y*scalar, a.z*scalar);
    }

    ///
    /// Computes the difference \f$ \vec{a}-\vec{b}\f$
    /// \note Creates a new vector instance
    ///
    template<class T>
    __device__ __host__ inline vec3<T> operator-(const vec3<T>& a, const vec3<T>& b) {
        return vec3<T>(a.x - b.x, a.y - b.y, a.z - b.z);
    }

    ///
    /// Computes the division \f$\frac{1}{\mathrm{scalar}} \vec{a}\f$
    /// \note Creates a new vector instance
    ///
    template<class T>
    __device__ __host__ inline vec3<T> operator/(const vec3<T>& a, T scalar) {
        return vec3<T>(a.x/scalar, a.y/scalar, a.z/scalar);
    }

    ///
    /// Computes the product \f$\vec{a}+\vec{b}\f$
    /// \note Creates a new vector instance.
    ///
    template<class T, class S>
    __device__ __host__ inline vec3<T> operator+(const vec3<T>& a, const vec3<S>& b) {
        return vec3<T>(a.x+b.x, a.y + b.y,  a.z + b.z);
    }



}
