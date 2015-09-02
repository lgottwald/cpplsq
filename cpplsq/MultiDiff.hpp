#ifndef _CPPLSQ_MULTI_DIFF_HPP_
#define _CPPLSQ_MULTI_DIFF_HPP_

#include <simd/alloc.hpp>
#include <cmath>
#include <iterator>
#include "AutoDiff.hpp"

namespace cpplsq
{

using namespace simd;

/**
 * Namespace for internal functions for
 * efficient memory management and
 * initialization of static data.
 */
namespace internal
{
extern std::size_t buffer_size;
extern std::size_t num_directions;

void *new_buffer();

void release_buffer( void * );

void free_all();
}

/**
 * Struct to count the number of simd temporaries
 * the expression requires to store in simd registers.
 */
template<typename T>
struct SimdTemps;

template<typename REAL>
class MultiDiff : public MultiDiffExpr<MultiDiff<REAL>>
{
public:
   struct Context
   {
      Context( std::size_t num_dir )
      {
         assert( internal::buffer_size == 0 );
         assert( internal::num_directions == 0 );
         internal::num_directions = next_size<REAL>( num_dir );
         internal::buffer_size = sizeof( REAL ) * internal::num_directions;
      }

      Context( const Context & ) = delete;
      Context( Context && other ) = delete;
      Context &operator=( const Context & ) = delete;
      Context &operator=( Context && other ) = delete;

      ~Context()
      {
         internal::free_all();
      }

   };

   MultiDiff()
   {
      dval = ( REAL * ) internal::new_buffer();
   }

   MultiDiff( REAL x ) : val( x )
   {
      dval = ( REAL * ) internal::new_buffer();
      setDiffValsZero();
   }

   MultiDiff( REAL x, std::size_t i ) : val( x )
   {
      dval = ( REAL * ) internal::new_buffer();
      setDiffValsZero();
      dval[i] = 1;
   }

   MultiDiff( const MultiDiff<REAL> &x ) : val( x.val )
   {
      dval = ( REAL * )  internal::new_buffer();

      for( std::size_t i = 0; i < internal::num_directions; i += pack_size<REAL>() )
         x.getDiffValues( i ).aligned_store( dval + i );
   }

   MultiDiff( MultiDiff<REAL> && x ) : val( x.val ), dval( x.dval )
   {
      x.dval = nullptr;
   }

   template<typename T>
   MultiDiff( const MultiDiffExpr<T> &x ) : val( x.getValue() )
   {
      dval = ( REAL * )internal::new_buffer();

      for( std::size_t i = 0; i < internal::num_directions; i += pack_size<REAL>() )
         x.getDiffValues( i ).aligned_store( dval + i );
   }

   //assignment
   MultiDiff<REAL> &operator=( REAL x )
   {
      val = x;
      setDiffValsZero();
      return *this;
   }

   MultiDiff<REAL> &operator=( const MultiDiff<REAL> &x )
   {
      val = x.val;

      for( std::size_t i = 0; i < internal::num_directions; i += pack_size<REAL>() )
         x.getDiffValues( i ).aligned_store( dval + i );

      return *this;
   }

   MultiDiff<REAL> &operator=( MultiDiff<REAL> && x )
   {
      std::swap( dval, x.dval );
      val = x.val;
      return *this;
   }

   template<typename T>
   MultiDiff<REAL> &operator=( const MultiDiffExpr<T> &x )
   {
      val = x.getValue();

      for( std::size_t i = 0; i < internal::num_directions; i += pack_size<REAL>() )
         x.getDiffValues( i ).aligned_store( dval + i );

      return *this;
   }

   //destruct
   ~MultiDiff()
   {
      internal::release_buffer( dval );
   }

   //access values / diff values

   REAL getValue() const
   {
      return val;
   }

   REAL getDiffValue( std::size_t i ) const
   {
      return dval[i];
   }

   pack<REAL> getDiffValues( std::size_t i ) const
   {
      assert( i < internal::num_directions );
      return aligned_load( dval + i );
   }

   const REAL *getDiffValues() const
   {
      return dval;
   }


   REAL *getDiffValues()
   {
      return dval;
   }

   void setIndependent( REAL v, std::size_t i )
   {
      val = v;
      setDiffValsZero();
      dval[i] = 1;
   }

   //arithmetic modifiers

   template<typename T>
   MultiDiff<REAL> &operator +=( const MultiDiffExpr<T> &x );

   template<typename T>
   MultiDiff<REAL> &operator -=( const MultiDiffExpr<T> &x );

   template<typename T>
   MultiDiff<REAL> &operator *=( const MultiDiffExpr<T> &x );

   template<typename T>
   MultiDiff<REAL> &operator /=( const MultiDiffExpr<T> &x );

private:
   void setDiffValsZero()
   {
      pack<REAL> z = zero<REAL>();

      REAL *const end = dval + internal::num_directions;

      for( REAL *i = dval; i != end; i += pack_size<REAL>() )
         z.aligned_store( i );
   }

   REAL val;
   REAL *dval;
};


template<typename REAL>
struct NumTypeTraits<MultiDiff<REAL>>
{
   using type = REAL;
};

template<typename REAL>
struct SimdTemps<MultiDiff<REAL>>
{
   constexpr static int value = 0;
};

/**
 * Declare given range of reals as independet variables and
 * return vector of MultiDiff objects to be used for calculations.
 */
template<typename RA_ITER, typename REAL = typename std::iterator_traits<RA_ITER>::value_type>
static std::vector<MultiDiff<REAL>> Independent( RA_ITER begin, RA_ITER end )
{
   static_assert( std::is_same< typename std::iterator_traits<RA_ITER>::iterator_category, std::random_access_iterator_tag>::value, "Requires a random access iterator" );

   std::vector<MultiDiff<REAL>> vars;
   vars.reserve( end - begin );

   for( RA_ITER i = begin; i != end; ++i )
      vars.emplace_back( *i, i - begin );

   return vars;
}

/**
 * Declare the given range of MultiDiff objects
 * as independent variables with the corresponding
 * values of the given value iterator.
 */
template < typename RA_ITER,
         typename VAL_ITER >
static void Independent( RA_ITER begin, RA_ITER end, VAL_ITER valiter )
{
   static_assert( std::is_same< typename std::iterator_traits<RA_ITER>::iterator_category, std::random_access_iterator_tag>::value, "Requires a random access iterator" );

   for( RA_ITER i = begin; i != end; ++i )
      i->setIndependent( *valiter++, i - begin );
}

/**
 * Expression template for adding two MultiDiff expressions
 */
template<typename A, typename B>
class MultiDiffPlus : public MultiDiffExpr<MultiDiffPlus<A, B>>
{
public:
   using REAL = NumType<MultiDiffPlus<A, B>>;

   MultiDiffPlus( const MultiDiffExpr<A> &a, const MultiDiffExpr<B> &b ) : a( a ), b( b ) {}

   REAL getValue() const
   {
      return a.getValue() + b.getValue();
   }

   pack<REAL> getDiffValues( std::size_t i ) const
   {
      return a.getDiffValues( i ) + b.getDiffValues( i );
   }

private:
   const MultiDiffExpr<A> &a;
   const MultiDiffExpr<B> &b;
};

template<typename A, typename B>
struct SimdTemps<MultiDiffPlus<A, B>>
{
   constexpr static int value = SimdTemps<A>::value + SimdTemps<B>::value;
};


template<typename A, typename B>
struct NumTypeTraits<MultiDiffPlus<A, B>>
{
   using type = decltype( NumType<A>( 0 ) + NumType<B>( 0 ) );
};

/**
 * Expression template for adding a constant to a MultiDiff expression
 */
template<typename T>
class ScalarMultiDiffPlus : public MultiDiffExpr<ScalarMultiDiffPlus<T>>
{
public:
   using REAL = NumType<ScalarMultiDiffPlus<T>>;

   ScalarMultiDiffPlus( REAL a, const MultiDiffExpr<T> &b ) : a( a ), b( b ) {}

   REAL getValue() const
   {
      return a + b.getValue();
   }

   pack<REAL> getDiffValues( std::size_t i ) const
   {
      return b.getDiffValues( i );
   }

private:
   REAL a;
   const MultiDiffExpr<T> &b;
};

template<typename T>
struct SimdTemps<ScalarMultiDiffPlus<T>>
{
   constexpr static int value = SimdTemps<T>::value;
};

template<typename T>
struct NumTypeTraits<ScalarMultiDiffPlus<T>>
{
   using type = NumType<T>;
};

/**
 * Expression template for substracting two MultiDiff expression
 */
template<typename A, typename B>
class MultiDiffMinus : public MultiDiffExpr<MultiDiffMinus<A, B>>
{
public:
   using REAL = NumType<MultiDiffMinus<A, B>>;

   MultiDiffMinus( const MultiDiffExpr<A> &a, const MultiDiffExpr<B> &b ) : a( a ), b( b ) {}

   REAL getValue() const
   {
      return a.getValue() - b.getValue();
   }

   pack<REAL> getDiffValues( std::size_t i ) const
   {
      return a.getDiffValues( i ) - b.getDiffValues( i );
   }

private:
   const MultiDiffExpr<A> &a;
   const MultiDiffExpr<B> &b;
};

template<typename A, typename B>
struct SimdTemps<MultiDiffMinus<A, B>>
{
   constexpr static int value = SimdTemps<A>::value + SimdTemps<B>::value;
};

template<typename A, typename B>
struct NumTypeTraits<MultiDiffMinus<A, B>>
{
   using type = decltype( NumType<A>( 0 ) - NumType<B>( 0 ) );
};

/**
 * Expression template for substracting a MultiDiff expression from a constant
 */
template<typename T>
class ScalarMultiDiffMinus : public MultiDiffExpr<ScalarMultiDiffMinus<T>>
{
public:
   using REAL = NumType<T>;

   ScalarMultiDiffMinus( REAL a, const MultiDiffExpr<T> &b ) : a( a ), b( b ) {}

   REAL getValue() const
   {
      return a - b.getValue();
   }

   pack<REAL> getDiffValues( std::size_t i ) const
   {
      return -b.getDiffValues( i );
   }

private:
   REAL a;
   const MultiDiffExpr<T> &b;
};

template<typename T>
struct SimdTemps<ScalarMultiDiffMinus<T>>
{
   constexpr static int value = SimdTemps<T>::value;
};

template<typename T>
struct NumTypeTraits<ScalarMultiDiffMinus<T>>
{
   using type = NumType<T>;
};

/**
 * Expression template for multiplying two MultiDiff expression
 */
template<typename A, typename B>
class MultiDiffMul : public MultiDiffExpr<MultiDiffMul<A, B>>
{
public:
   using REAL = NumType<MultiDiffMul<A, B>>;

   MultiDiffMul( const MultiDiffExpr<A> &a, const MultiDiffExpr<B> &b ) : aval( a.getValue() ), bval( b.getValue() ), a( a ), b( b ) {}

   REAL getValue() const
   {
      return aval[0] * bval[0];
   }

   pack<REAL> getDiffValues( std::size_t i ) const
   {
      return bval * a.getDiffValues( i ) + aval * b.getDiffValues( i );
   }

private:
   pack<REAL> aval;
   pack<REAL> bval;
   const MultiDiffExpr<A> &a;
   const MultiDiffExpr<B> &b;
};


template<typename A, typename B>
struct SimdTemps<MultiDiffMul<A, B>>
{
   static constexpr int value = SimdTemps<A>::value + SimdTemps<B>::value + 2;
};

template<typename A, typename B>
struct NumTypeTraits<MultiDiffMul<A, B>>
{
   using type = decltype( NumType<A>( 0 ) * NumType<B>( 0 ) );
};

/**
 * Expression template for multiplying a MultiDiff expression with a constant
 */
template<typename T>
class ScalarMultiDiffMul : public MultiDiffExpr<ScalarMultiDiffMul<T>>
{
public:
   using REAL = NumType<ScalarMultiDiffMul<T>>;

   ScalarMultiDiffMul( REAL a, const MultiDiffExpr<T> &b ) : aval( a ), b( b ) {}

   REAL getValue() const
   {
      return aval[0] * b.getValue();
   }

   pack<REAL> getDiffValues( std::size_t i ) const
   {
      return aval * b.getDiffValues( i );
   }

private:
   pack<REAL> aval;
   const MultiDiffExpr<T> &b;
};

template<typename T>
struct SimdTemps<ScalarMultiDiffMul<T>>
{
   static constexpr int value = SimdTemps<T>::value + 1;
};

template<typename T>
struct NumTypeTraits<ScalarMultiDiffMul<T>>
{
   using type = NumType<T>;
};

/**
 * Expression template for division of two MultiDiff expressions
 */
template<typename A, typename B>
class MultiDiffDiv : public MultiDiffExpr<MultiDiffDiv<A, B>>
{
public:
   using REAL = NumType<MultiDiffDiv<A, B>>;

   MultiDiffDiv( const MultiDiffExpr<A> &a, const MultiDiffExpr<B> &b ) : aval( a.getValue() ), bval( b.getValue() ), a( a ), b( b )
   {
      b2val = pack<REAL>( bval[0] * bval[0] );
   }

   REAL getValue() const
   {
      return aval[0] / bval[0];
   }

   pack<REAL> getDiffValues( std::size_t i ) const
   {
      return ( bval * a.getDiffValues( i ) - aval * b.getDiffValues( i ) ) / b2val;
   }

private:
   pack<REAL> aval;
   pack<REAL> bval;
   pack<REAL> b2val;
   const MultiDiffExpr<A> &a;
   const MultiDiffExpr<B> &b;
};

template<typename A, typename B>
struct SimdTemps<MultiDiffDiv<A, B>>
{
   static constexpr int value = SimdTemps<A>::value + SimdTemps<B>::value + 3;
};

template<typename A, typename B>
struct NumTypeTraits<MultiDiffDiv<A, B>>
{
   using type = decltype( NumType<A>( 0 ) / NumType<B>( 1. ) );
};

/**
 * Expression template for division of a constant by a MultiDiff expression
 */
template<typename T>
class ScalarMultiDiffDiv : public MultiDiffExpr<ScalarMultiDiffDiv<T>>
{
public:
   using REAL = NumType<ScalarMultiDiffDiv<T>>;

   ScalarMultiDiffDiv( REAL a, const MultiDiffExpr<T> &b ) : bval( b.getValue() ), aval( a ), b2val( bval *bval ) {}

   REAL getValue() const
   {
      return aval[0] / bval;
   }

   pack<REAL> getDiffValues( std::size_t i ) const
   {
      return - aval * b.getDiffValues( i ) / b2val;
   }

private:
   REAL bval;
   pack<REAL> aval;
   pack<REAL> b2val;
   const MultiDiffExpr<T> &b;
};

template<typename T>
struct SimdTemps<ScalarMultiDiffDiv<T>>
{
   static constexpr int value = SimdTemps<T>::value + 2;
};

template<typename T>
struct NumTypeTraits<ScalarMultiDiffDiv<T>>
{
   using type = NumType<T>;
};

/**
 * Expression template for division of a MultiDiff expression by a constant
 */
template<typename T>
class MultiDiffScalarDiv : public MultiDiffExpr<MultiDiffScalarDiv<T>>
{
public:
   using REAL = NumType<MultiDiffScalarDiv<T>>;

   MultiDiffScalarDiv( const MultiDiffExpr<T> &a, REAL b ) : a( a ), bval( b ) {}

   REAL getValue() const
   {
      return a.getValue() / bval[0];
   }


   pack<REAL> getDiffValues( std::size_t i ) const
   {
      return a.getDiffValues( i ) / bval;
   }

private:
   const MultiDiffExpr<T> &a;
   pack<REAL> bval;
};

template<typename T>
struct SimdTemps<MultiDiffScalarDiv<T>>
{
   static constexpr int value = SimdTemps<T>::value + 1;
};

template<typename T>
struct NumTypeTraits<MultiDiffScalarDiv<T>>
{
   using type = NumType<T>;
};

/**
 * Expression template for computing exponential function of a MultiDiff expression
 */
template<typename T>
class MultiDiffExp : public MultiDiffExpr<MultiDiffExp<T>>
{
public:
   using REAL = NumType<MultiDiffExp<T>>;

   MultiDiffExp( const MultiDiffExpr<T> &x ) : x( x )
   {
      val = pack<REAL>( std::exp( x.getValue() ) );
   }

   REAL getValue() const
   {
      return val[0];
   }

   pack<REAL> getDiffValues( std::size_t i ) const
   {
      return val * x.getDiffValues( i );
   }

private:
   pack<REAL> val;
   const MultiDiffExpr<T> &x;
};

template<typename T>
struct SimdTemps<MultiDiffExp<T>>
{
   static constexpr int value = SimdTemps<T>::value + 1;
};

template<typename T>
struct NumTypeTraits<MultiDiffExp<T>>
{
   using type = NumType<T>;
};

/**
 * Expression template for negating a MultiDiff expression
 */
template<typename T>
class MultiDiffNeg : public MultiDiffExpr<MultiDiffNeg<T>>
{
public:
   using REAL = NumType<MultiDiffNeg<T>>;

   MultiDiffNeg( const MultiDiffExpr<T> &x ) : x( x ) {}

   REAL getValue() const
   {
      return -x.getValue();
   }

   pack<REAL> getDiffValues( std::size_t i ) const
   {
      return -x.getDiffValues( i );
   }

private:
   const MultiDiffExpr<T> &x;
};

template<typename T>
struct SimdTemps<MultiDiffNeg<T>>
{
   constexpr static int value = SimdTemps<T>::value;
};

template<typename T>
struct NumTypeTraits<MultiDiffNeg<T>>
{
   using type = NumType<T>;
};


/**
 * Decide if a proxy object is used or if the current subexpression is stored in a temporary
 * to reduce register pressure based on the simd temporaries that are required for the expression.
 */
template < typename T, int MAX_TMPS, bool CREATE_VAL = ( SimdTemps<T>::value > MAX_TMPS ) >
struct ReturnTypeTraits;

template<typename T, int MAX_TMPS>
struct ReturnTypeTraits<T, MAX_TMPS, true>
{
   using type = MultiDiff<NumType<T>>;
};

template<typename T, int MAX_TMPS>
struct ReturnTypeTraits<T, MAX_TMPS, false>
{
   using type = T;
};

template<typename T>
using ReturnType = typename ReturnTypeTraits<T, 8>::type;

template<typename T>
ReturnType<MultiDiffExp<T>> exp( const MultiDiffExpr<T> &x )
{
   return MultiDiffExp<T>( x );
}

template<typename T>
ReturnType<MultiDiffNeg<T>> operator-( const MultiDiffExpr<T> &x )
{
   return MultiDiffNeg<T>( x );
}

template<typename A, typename B>
ReturnType<MultiDiffPlus<A, B>> operator+( const MultiDiffExpr<A> &a, const MultiDiffExpr<B> &b )
{
   return MultiDiffPlus<A, B>( a, b );
}

template<typename A, typename B>
ReturnType<MultiDiffMinus<A, B>> operator-( const MultiDiffExpr<A> &a, const MultiDiffExpr<B> &b )
{
   return MultiDiffMinus<A, B>( a, b );
}

template<typename A, typename B>
ReturnType<MultiDiffMul<A, B>> operator*( const MultiDiffExpr<A> &a, const MultiDiffExpr<B> &b )
{
   return MultiDiffMul<A, B>( a, b );
}

template<typename A, typename B>
ReturnType<MultiDiffDiv<A, B>> operator/( const MultiDiffExpr<A> &a, const MultiDiffExpr<B> &b )
{
   return MultiDiffDiv<A, B>( a, b );
}

template<typename T>
ReturnType<ScalarMultiDiffPlus<T>> operator+( const MultiDiffExpr<T> &a, NumType<T> b )
{
   return ScalarMultiDiffPlus<T>( b, a );
}

template<typename T>
ReturnType<ScalarMultiDiffPlus<T>> operator-( const MultiDiffExpr<T> &a, NumType<T> b )
{
   return ScalarMultiDiffPlus<T>( -b, a );
}

template<typename T>
ReturnType<ScalarMultiDiffMul<T>> operator*( const MultiDiffExpr<T> &a, NumType<T> b )
{
   return ScalarMultiDiffMul<T>( b, a );
}

template<typename T>
ReturnType<MultiDiffScalarDiv<T>> operator/( const MultiDiffExpr<T> &a, NumType<T> b )
{
   return MultiDiffScalarDiv<T>( a, b );
}

template<typename T>
ReturnType<ScalarMultiDiffPlus<T>> operator+( const NumType<T> a, const MultiDiffExpr<T> &b )
{
   return ScalarMultiDiffPlus<T>( a, b );
}

template<typename T>
ReturnType<ScalarMultiDiffMinus<T>> operator-( const NumType<T> a, const MultiDiffExpr<T> &b )
{
   return ScalarMultiDiffMinus<T>( a, b );
}

template<typename T>
ReturnType<ScalarMultiDiffMul<T>> operator*( const NumType<T> a, const MultiDiffExpr<T> &b )
{
   return ScalarMultiDiffMul<T>( a, b );
}

template<typename T>
ReturnType<ScalarMultiDiffDiv<T>> operator/( const NumType<T> a, const MultiDiffExpr<T> &b )
{
   return ScalarMultiDiffDiv<T>( a, b );
}


//RELATIONAL =======================================================

template<typename A, typename B>
bool operator<( const MultiDiffExpr<A> &a, const MultiDiffExpr<B> &b )
{
   return a.getValue() < b.getValue();
}

template<typename A, typename B>
bool operator>( const MultiDiffExpr<A> &a, const MultiDiffExpr<B> &b )
{
   return a.getValue() > b.getValue();
}

template<typename A, typename B>
bool operator<=( const MultiDiffExpr<A> &a, const MultiDiffExpr<B> &b )
{
   return a.getValue() <= b.getValue();
}

template<typename A, typename B>
bool operator>=( const MultiDiffExpr<A> &a, const MultiDiffExpr<B> &b )
{
   return a.getValue() >= b.getValue();
}

template<typename A, typename B>
bool operator==( const MultiDiffExpr<A> &a, const MultiDiffExpr<B> &b )
{
   return a.getValue() == b.getValue();
}

template<typename A, typename B>
bool operator!=( const MultiDiffExpr<A> &a, const MultiDiffExpr<B> &b )
{
   return a.getValue() != b.getValue();
}



template<typename T>
bool operator<( const MultiDiffExpr<T> &a, NumType<T> b )
{
   return a.getValue() < b;
}

template<typename T>
bool operator>( const MultiDiffExpr<T> &a, NumType<T> b )
{
   return a.getValue() > b;
}

template<typename T>
bool operator<=( const MultiDiffExpr<T> &a, NumType<T> b )
{
   return a.getValue() <= b;
}

template<typename T>
bool operator>=( const MultiDiffExpr<T> &a, NumType<T> b )
{
   return a.getValue() >= b;
}

template<typename T>
bool operator==( const MultiDiffExpr<T> &a, NumType<T> b )
{
   return a.getValue() == b;
}

template<typename T>
bool operator!=( const MultiDiffExpr<T> &a, NumType<T> b )
{
   return a.getValue() != b;
}



template<typename T>
bool operator<( NumType<T> a, const MultiDiffExpr<T> &b )
{
   return a < b.getValue();
}

template<typename T>
bool operator>( NumType<T> a, const MultiDiffExpr<T> &b )
{
   return a > b.getValue();
}

template<typename T>
bool operator<=( NumType<T> a, const MultiDiffExpr<T> &b )
{
   return a <= b.getValue();
}

template<typename T>
bool operator>=( NumType<T> a, const MultiDiffExpr<T> &b )
{
   return a >= b.getValue();
}

template<typename T>
bool operator==( NumType<T> a, const MultiDiffExpr<T> &b )
{
   return a == b.getValue();
}

template<typename T>
bool operator!=( NumType<T> a, const MultiDiffExpr<T> &b )
{
   return a != b.getValue();
}

template<typename REAL>
template<typename T>
MultiDiff<REAL> &MultiDiff<REAL>::operator +=( const MultiDiffExpr<T> &x )
{
   *this = MultiDiffPlus<MultiDiff<REAL>, T>( *this, x );
   return *this;
}

template<typename REAL>
template<typename T>
MultiDiff<REAL> &MultiDiff<REAL>::operator -=( const MultiDiffExpr<T> &x )
{
   *this = MultiDiffMinus<MultiDiff<REAL>, T>( *this, x );
   return *this;
}

template<typename REAL>
template<typename T>
MultiDiff<REAL> &MultiDiff<REAL>::operator *=( const MultiDiffExpr<T> &x )
{
   *this = MultiDiffMul<MultiDiff<REAL>, T>( *this , x );
   return *this;
}

template<typename REAL>
template<typename T>
MultiDiff<REAL> &MultiDiff<REAL>::operator /=( const MultiDiffExpr<T> &x )
{
   *this = MultiDiffDiv<MultiDiff<REAL>, T>( *this, x );
   return *this;
}

} //tlfd



#endif
