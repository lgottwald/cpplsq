#ifndef _CPPLSQ_LINE_SEARCH_HPP_
#define _CPPLSQ_LINE_SEARCH_HPP_

#include <cmath>
#include <limits>
#include "SingleDiff.hpp"

namespace cpplsq
{

/**
 * Perform line_search for the given univariate function starting
 * at the given point.
 *
 * \param f0     starting point
 * \param f      function accepting a single real as argument and
 *               returning a SingleDiff object containing the value
 *               and the derivative for the given value.
 * \param alpha  real value to store the step length. Initial value
 *               when the function is called will be used as the
 *               initial step length. On termination the value will
 *               be a step length satisfying the weak wolfe conditions
 *               if the line search succeeded or the last tested step
 *               length.
 *
 * \return       true if the line search succeeded and found a step
 *               length satisfying the weak wolfe conditions.
 *
 */
template<typename REAL, typename FUNC>
bool line_search( SingleDiff<REAL> f0, const FUNC &f, REAL &alpha )
{
   constexpr static REAL c1 = 1e-4;
   constexpr static REAL c2 = 0.9;
   constexpr static int LINESEARCH_MAXITER = std::ceil( -std::log2( std::pow( std::numeric_limits<REAL>::epsilon(), 2. / 3. ) ) );
   constexpr static REAL inf = std::numeric_limits<REAL>::max();

   REAL lo = 0;
   REAL up = inf;

   for( int l = 0; l < LINESEARCH_MAXITER; ++l )
   {
      SingleDiff<REAL> fval = f( alpha );

      if( !std::isfinite( fval.getValue() ) || fval.getValue() > f0.getValue() + c1 * alpha * f0.getDiffValue() )
         up = alpha;
      else if( fval.getDiffValue() < c2 * f0.getDiffValue() )
         lo = alpha;
      else
         return true;

      if( up < inf )
         alpha = ( up + lo ) / 2;
      else
         alpha *= 2;
   }

   return false;
}


}
#endif
