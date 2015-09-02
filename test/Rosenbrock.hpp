#ifndef _CPPLSQ_TEST_ROSENBROCK_HPP_
#define _CPPLSQ_TEST_ROSENBROCK_HPP_

#include <vector>
#include <type_traits>

template<typename REAL>
REAL rosen_brock( const REAL *x, std::size_t N )
{
   REAL val = 0;

   auto sqr = []( REAL x )
   {
      return x * x;
   };

   for( std::size_t i = 0; i < N - 1; ++i )
   {
      val += sqr( 1 - x[i] ) + 100 * sqr( x[i + 1] - sqr( x[i] ) );
   }

   return val;
}

template<typename REAL>
std::vector<REAL>  rosen_brock_deriv( const std::vector<REAL> &x )
{
   std::vector<REAL> y( x.size(), REAL( 0 ) );

   auto sqr = []( REAL x )
   {
      return x * x;
   };

   //(d)/(dx)((1-x)^2+100 (a-x^2)^2) = 2 (-200 a x+200 x^3+x-1)
   //(d)/(dx)((1-a)^2+100 (x-a^2)^2) = 200 (x-a^2)
   for( std::size_t i = 0; i < x.size() - 1; ++i )
   {
      auto xi2 = sqr( x[i] );
      y[i] += 2 * ( -200 * x[i] * x[i + 1] + 200 * xi2 * x[i] + x[i] - 1 );
      y[i + 1] += 200 * ( x[i + 1] - xi2 );
   }

   return y;
}

#endif
