#ifndef _CPPLSQ_AUTO_DIFF_HPP_
#define _CPPLSQ_AUTO_DIFF_HPP_

#include <simd/pack.hpp>
#include <type_traits>

namespace cpplsq
{

using namespace simd;

/**
 * Traits to specify the type for floating point numbers used by the
 * given AutoDiff type
 */
template<typename T>
struct NumTypeTraits
{
   using type = void;
};

template<typename T>
using NumType = typename NumTypeTraits<T>::type;

/**
 * Crtp base class for MultiDiff types.
 */
template<typename IMPL>
struct MultiDiffExpr
{
   static_assert( !std::is_void<NumType<IMPL>>::value, "NumTypeTraits not specialized" );
   using REAL = NumType<IMPL>;

   REAL getValue() const
   {
      return static_cast<const IMPL *>( this )->getValue();
   }

   pack<REAL> getDiffValues( std::size_t i ) const
   {
      return static_cast<const IMPL *>( this )->getDiffValues( i );
   }

};

/**
 * Crtp base class for SingleDiff types.
 */
template<typename IMPL>
struct SingleDiffExpr
{
   static_assert( !std::is_void<NumType<IMPL>>::value, "NumTypeTraits not specialized" );
   using REAL = NumType<IMPL>;

   REAL getValue() const
   {
      return static_cast<const IMPL *>( this )->getValue();
   }

   REAL getDiffValue() const
   {
      return static_cast<const IMPL *>( this )->getDiffValue();
   }
};

/**
 * Helper function to check if given type is a MultiDiff type
 */
template<typename T>
constexpr bool is_multi_diff_type()
{
   return std::is_base_of<MultiDiffExpr<T>, T>::value;
}

/**
 * Helper function to check if given type is a SingleDiff type
 */
template<typename T>
constexpr bool is_single_diff_type()
{
   return std::is_base_of<SingleDiffExpr<T>, T>::value;
}

/**
 * Helper function to check if given type is a SingleDiff or MultiDiff type
 */
template<typename T>
constexpr bool is_diff_type()
{
   return is_multi_diff_type<T>() || is_single_diff_type<T>();
}

/**
 * MultiDiff class for computing derivatives in multiple
 * independent variables using operator overloading
 * and forward differentiation.
 */
template <typename REAL>
class MultiDiff;

/**
 * SingleDiff type for computing the derivative in a
 * single variable using operator overloading.
 */
template <typename REAL>
class SingleDiff;

/**
 * Gives the type to store the value of the given
 * type. For MultiDiff and SingleDiff expression types
 * this will evaluate to the MultiDiff and SingleDiff classes
 * with the corresponding number type. For other types
 * just evaluates to the given type.
 * This can be used to accept an expression template
 * of MultiDiff/SingleDiff as parameter and return
 * a stored value instead of an expression.
 */
template<typename T>
using ValueType = typename std::conditional < is_multi_diff_type<T>(),
      MultiDiff<NumType<T>>,
      typename std::conditional<is_single_diff_type<T>(), SingleDiff<NumType<T>>, T>::type >::type;

} //cpplsq

#endif

