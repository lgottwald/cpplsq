#ifndef _CPPLSQ_SINGLE_DIFF_HPP_
#define _CPPLSQ_SINGLE_DIFF_HPP_

#include <cmath>
#include "AutoDiff.hpp"

namespace cpplsq
{


template<typename REAL>
class SingleDiff : public SingleDiffExpr<SingleDiff<REAL>>
{
public:
   SingleDiff() = default;

   SingleDiff( const SingleDiff<REAL> &other ) = default;

   SingleDiff( SingleDiff<REAL> && ) = default;

   SingleDiff<REAL> &operator=( const  SingleDiff<REAL> & ) = default;

   SingleDiff<REAL> &operator=( SingleDiff<REAL> && ) = default;

   template<typename T>
   SingleDiff( const SingleDiffExpr<T> &a ) : val( a.getValue() ), diffval( a.getDiffValue() ) {}

   SingleDiff( REAL val ) : val( val ), diffval( 0 ) {}

   SingleDiff( REAL val, REAL dval ) : val( val ), diffval( dval ) {}

   REAL getValue() const
   {
      return val;
   }

   REAL getDiffValue() const
   {
      return diffval;
   }

   SingleDiff<REAL> &operator=( REAL x )
   {
      val = x;
      diffval = 0;
      return *this;
   }

   void operator, ( REAL x )
   {
      diffval = x;
   }

   template<typename T>
   SingleDiff<REAL> &operator=( const SingleDiffExpr<T> &x )
   {
      val = x.getValue();
      diffval = x.getDiffValue();
      return *this;
   }

   template<typename T>
   SingleDiff<REAL> &operator +=( const SingleDiffExpr<T> &x )
   {
      *this = *this + x;
      return *this;
   }

   template<typename T>
   SingleDiff<REAL> &operator -=( const SingleDiffExpr<T> &x )
   {
      *this = *this - x;
      return *this;
   }

   template<typename T>
   SingleDiff<REAL> &operator *=( const SingleDiffExpr<T> &x )
   {
      *this = *this * x;
      return *this;
   }

   template<typename T>
   SingleDiff<REAL> &operator /=( const SingleDiffExpr<T> &x )
   {
      *this = *this / x;
      return *this;
   }

private:
   REAL val;
   REAL diffval;
};

template<typename REAL>
struct NumTypeTraits<SingleDiff<REAL>>
{
   using type = REAL;
};


template<typename A, typename B>
class SingleDiffPlus : public SingleDiffExpr<SingleDiffPlus<A, B>>
{
public:
   SingleDiffPlus( const SingleDiffExpr<A> &a, const SingleDiffExpr<B> &b ) : a( a ), b( b ) {}

   using REAL = NumType<SingleDiffPlus<A, B>>;

   REAL getValue() const
   {
      return a.getValue() + b.getValue();
   }

   REAL getDiffValue() const
   {
      return a.getDiffValue() + b.getDiffValue();
   }

private:
   const SingleDiffExpr<A> &a;
   const SingleDiffExpr<B> &b;
};


template<typename A, typename B>
struct NumTypeTraits<SingleDiffPlus<A, B>>
{
   using type = decltype( NumType<A>( 0 ) + NumType<B>( 0 ) );
};

template<typename T>
class ScalarSingleDiffPlus : public SingleDiffExpr<ScalarSingleDiffPlus<T>>
{
public:
   using REAL = NumType<ScalarSingleDiffPlus<T>>;

   ScalarSingleDiffPlus( REAL a, const SingleDiffExpr<T> &b ) : aval( a ), b( b ) {}

   REAL getValue() const
   {
      return aval + b.getValue();
   }

   REAL getDiffValue() const
   {
      return b.getDiffValue();
   }

private:
   REAL aval;
   const SingleDiffExpr<T> &b;
};

template<typename T>
struct NumTypeTraits<ScalarSingleDiffPlus<T>>
{
   using type = NumType<T>;
};

template<typename A, typename B>
class SingleDiffMinus : public SingleDiffExpr<SingleDiffMinus<A, B>>
{
public:
   SingleDiffMinus( const SingleDiffExpr<A> &a, const SingleDiffExpr<B> &b ) : a( a ), b( b ) {}

   using REAL = NumType<SingleDiffMinus<A, B>>;

   REAL getValue() const
   {
      return a.getValue() - b.getValue();
   }

   REAL getDiffValue() const
   {
      return a.getDiffValue() - b.getDiffValue();
   }

private:
   const SingleDiffExpr<A> &a;
   const SingleDiffExpr<B> &b;
};

template<typename A, typename B>
struct NumTypeTraits<SingleDiffMinus<A, B>>
{
   using type = decltype( NumType<A>( 0 ) - NumType<B>( 0 ) );
};

template<typename T>
class ScalarSingleDiffMinus : public SingleDiffExpr<ScalarSingleDiffMinus<T>>
{
public:
   using REAL = NumType<ScalarSingleDiffMinus<T>>;

   ScalarSingleDiffMinus( REAL a, const SingleDiffExpr<T> &b ) : aval( a ), b( b ) {}

   REAL getValue() const
   {
      return aval - b.getValue();
   }

   REAL getDiffValue() const
   {
      return -b.getDiffValue();
   }

private:
   REAL aval;
   const SingleDiffExpr<T> &b;
};

template<typename T>
struct NumTypeTraits<ScalarSingleDiffMinus<T>>
{
   using type = NumType<T>;
};

template<typename A, typename B>
class SingleDiffDiv : public SingleDiffExpr<SingleDiffDiv<A, B>>
{
public:
   SingleDiffDiv( const SingleDiffExpr<A> &a, const SingleDiffExpr<B> &b ) : a( a ), b( b )
   {
      aval = a.getValue();
      bval = b.getValue();
   }

   using REAL = NumType<SingleDiffDiv<A, B>>;

   REAL getValue() const
   {
      return aval / bval;
   }

   REAL getDiffValue() const
   {
      return ( bval * a.getDiffValue() - aval * b.getDiffValue() ) / ( bval * bval );
   }

private:
   REAL aval;
   REAL bval;
   const SingleDiffExpr<A> &a;
   const SingleDiffExpr<B> &b;
};

template<typename A, typename B>
struct NumTypeTraits<SingleDiffDiv<A, B>>
{
   using type = decltype( NumType<A>( 0 ) / NumType<B>( 0 ) );
};

template<typename T>
class SingleDiffScalarDiv : public SingleDiffExpr<SingleDiffScalarDiv<T>>
{
public:
   using REAL = NumType<SingleDiffScalarDiv<T>>;

   SingleDiffScalarDiv( const SingleDiffExpr<T> &a, REAL b ) : a( a ), bval( b ) {}

   REAL getValue() const
   {
      return a.getValue() / bval;
   }


   REAL getDiffValue( ) const
   {
      return a.getDiffValue() / bval;
   }

private:
   const SingleDiffExpr<T> &a;
   REAL bval;
};

template<typename T>
struct NumTypeTraits<SingleDiffScalarDiv<T>>
{
   using type = NumType<T>;
};

template<typename T>
class ScalarSingleDiffDiv : public SingleDiffExpr<ScalarSingleDiffDiv<T>>
{
public:
   using REAL = NumType<ScalarSingleDiffDiv<T>>;

   ScalarSingleDiffDiv( REAL a, const SingleDiffExpr<T> &b ) : bval( b.getValue() ), aval( a ), b( b ) {}

   REAL getValue() const
   {
      return aval / bval;
   }

   REAL getDiffValue() const
   {
      return - aval * b.getDiffValue() / ( bval * bval );
   }

private:
   REAL bval;
   REAL aval;
   const SingleDiffExpr<T> &b;
};

template<typename T>
struct NumTypeTraits<ScalarSingleDiffDiv<T>>
{
   using type = NumType<T>;
};


template<typename A, typename B>
class SingleDiffMul : public SingleDiffExpr<SingleDiffMul<A, B>>
{
public:
   SingleDiffMul( const SingleDiffExpr<A> &a, const SingleDiffExpr<B> &b ) : a( a ), b( b )
   {
      aval = a.getValue();
      bval = b.getValue();
   }

   using REAL = NumType<SingleDiffMul<A, B>>;

   REAL getValue() const
   {
      return aval * bval;
   }

   REAL getDiffValue() const
   {
      return aval * b.getDiffValue() + bval * a.getDiffValue();
   }

private:
   REAL aval;
   REAL bval;
   const SingleDiffExpr<A> &a;
   const SingleDiffExpr<B> &b;
};

template<typename A, typename B>
struct NumTypeTraits<SingleDiffMul<A, B>>
{
   using type = decltype( NumType<A>( 0 ) * NumType<B>( 0 ) );
};

template<typename T>
class ScalarSingleDiffMul : public SingleDiffExpr<ScalarSingleDiffMul<T>>
{
public:
   using REAL = NumType<ScalarSingleDiffMul<T>>;

   ScalarSingleDiffMul( REAL a, const SingleDiffExpr<T> &b ) : aval( a ), b( b ) {}

   REAL getValue() const
   {
      return aval * b.getValue();
   }

   REAL getDiffValue() const
   {
      return aval * b.getDiffValue();
   }

private:
   REAL aval;
   const SingleDiffExpr<T> &b;
};

template<typename T>
struct NumTypeTraits<ScalarSingleDiffMul<T>>
{
   using type = NumType<T>;
};

template<typename T>
class SingleDiffExp : public SingleDiffExpr<SingleDiffExp<T>>
{
public:
   using REAL = NumType<SingleDiffExp<T>>;

   SingleDiffExp( const SingleDiffExpr<T> &x ) : x( x )
   {
      val = std::exp( x.getValue() );
   }

   REAL getValue() const
   {
      return val;
   }

   REAL getDiffValue() const
   {
      return val * x.getDiffValue();
   }

private:
   REAL val;
   const SingleDiffExpr<T> &x;
};

template<typename T>
struct NumTypeTraits<SingleDiffExp<T>>
{
   using type = NumType<T>;
};

template<typename T>
class SingleDiffNeg : public SingleDiffExpr<SingleDiffNeg<T>>
{
public:
   using REAL = NumType<SingleDiffNeg<T>>;

   SingleDiffNeg( const SingleDiffExpr<T> &x ) : x( x ) {}

   REAL getValue() const
   {
      return -x.getValue();
   }

   REAL getDiffValue() const
   {
      return -x.getDiffValue();
   }

private:
   const SingleDiffExpr<T> &x;
};

template<typename T>
struct NumTypeTraits<SingleDiffNeg<T>>
{
   using type = NumType<T>;
};

template<typename T>
SingleDiffExp<T> exp( const SingleDiffExpr<T> &x )
{
   return SingleDiffExp<T>( x );
}

template<typename T>
SingleDiffNeg<T> operator-( const SingleDiffExpr<T> &x )
{
   return SingleDiffNeg<T>( x );
}

template<typename A, typename B>
SingleDiffPlus<A, B> operator+( const SingleDiffExpr<A> &a, const SingleDiffExpr<B> &b )
{
   return SingleDiffPlus<A, B>( a, b );
}

template<typename A, typename B>
SingleDiffMinus<A, B> operator-( const SingleDiffExpr<A> &a, const SingleDiffExpr<B> &b )
{
   return SingleDiffMinus<A, B>( a, b );
}

template<typename A, typename B>
SingleDiffMul<A, B> operator*( const SingleDiffExpr<A> &a, const SingleDiffExpr<B> &b )
{
   return SingleDiffMul<A, B>( a, b );
}

template<typename A, typename B>
SingleDiffDiv<A, B> operator/( const SingleDiffExpr<A> &a, const SingleDiffExpr<B> &b )
{
   return SingleDiffDiv<A, B>( a, b );
}

template<typename T>
ScalarSingleDiffPlus<T> operator+( const SingleDiffExpr<T> &a, NumType<T> b )
{
   return ScalarSingleDiffPlus<T>( b, a );
}

template<typename T>
ScalarSingleDiffPlus<T> operator-( const SingleDiffExpr<T> &a, NumType<T> b )
{
   return ScalarSingleDiffPlus<T>( -b, a );
}

template<typename T>
ScalarSingleDiffMul<T> operator*( const SingleDiffExpr<T> &a, NumType<T> b )
{
   return ScalarSingleDiffMul<T>( b, a );
}

template<typename T>
SingleDiffScalarDiv<T> operator/( const SingleDiffExpr<T> &a, NumType<T> b )
{
   return SingleDiffScalarDiv<T>( a, b );
}

template<typename T>
ScalarSingleDiffPlus<T> operator+( const NumType<T> a, const SingleDiffExpr<T> &b )
{
   return ScalarSingleDiffPlus<T>( a, b );
}

template<typename T>
ScalarSingleDiffMinus<T> operator-( const NumType<T> a, const SingleDiffExpr<T> &b )
{
   return ScalarSingleDiffMinus<T>( a, b );
}

template<typename T>
ScalarSingleDiffMul<T> operator*( const NumType<T> a, const SingleDiffExpr<T> &b )
{
   return ScalarSingleDiffMul<T>( a, b );
}

template<typename T>
ScalarSingleDiffDiv<T> operator/( const NumType<T> a, const SingleDiffExpr<T> &b )
{
   return ScalarSingleDiffDiv<T>( a, b );
}


//RELATIONAL =======================================================

template<typename A, typename B>
bool operator<( const SingleDiffExpr<A> &a, const SingleDiffExpr<B> &b )
{
   return a.getValue() < b.getValue();
}

template<typename A, typename B>
bool operator>( const SingleDiffExpr<A> &a, const SingleDiffExpr<B> &b )
{
   return a.getValue() > b.getValue();
}

template<typename A, typename B>
bool operator<=( const SingleDiffExpr<A> &a, const SingleDiffExpr<B> &b )
{
   return a.getValue() <= b.getValue();
}

template<typename A, typename B>
bool operator>=( const SingleDiffExpr<A> &a, const SingleDiffExpr<B> &b )
{
   return a.getValue() >= b.getValue();
}

template<typename A, typename B>
bool operator==( const SingleDiffExpr<A> &a, const SingleDiffExpr<B> &b )
{
   return a.getValue() == b.getValue();
}

template<typename A, typename B>
bool operator!=( const SingleDiffExpr<A> &a, const SingleDiffExpr<B> &b )
{
   return a.getValue() != b.getValue();
}



template<typename T>
bool operator<( const SingleDiffExpr<T> &a, NumType<T> b )
{
   return a.getValue() < b;
}

template<typename T>
bool operator>( const SingleDiffExpr<T> &a, NumType<T> b )
{
   return a.getValue() > b;
}

template<typename T>
bool operator<=( const SingleDiffExpr<T> &a, NumType<T> b )
{
   return a.getValue() <= b;
}

template<typename T>
bool operator>=( const SingleDiffExpr<T> &a, NumType<T> b )
{
   return a.getValue() >= b;
}

template<typename T>
bool operator==( const SingleDiffExpr<T> &a, NumType<T> b )
{
   return a.getValue() == b;
}

template<typename T>
bool operator!=( const SingleDiffExpr<T> &a, NumType<T> b )
{
   return a.getValue() != b;
}



template<typename T>
bool operator<( NumType<T> a, const SingleDiffExpr<T> &b )
{
   return a < b.getValue();
}

template<typename T>
bool operator>( NumType<T> a, const SingleDiffExpr<T> &b )
{
   return a > b.getValue();
}

template<typename T>
bool operator<=( NumType<T> a, const SingleDiffExpr<T> &b )
{
   return a <= b.getValue();
}

template<typename T>
bool operator>=( NumType<T> a, const SingleDiffExpr<T> &b )
{
   return a >= b.getValue();
}

template<typename T>
bool operator==( NumType<T> a, const SingleDiffExpr<T> &b )
{
   return a == b.getValue();
}

template<typename T>
bool operator!=( NumType<T> a, const SingleDiffExpr<T> &b )
{
   return a != b.getValue();
}


} //cpplsq

#endif
