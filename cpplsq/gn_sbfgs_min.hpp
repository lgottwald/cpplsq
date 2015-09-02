#ifndef _CPPLSQ_GN_SBFGS_MIN_HPP_
#define _CPPLSQ_GN_SBFGS_MIN_HPP_

#include <memory>
#include <functional>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <simd/alloc.hpp>
#include <simd/pack.hpp>
#include "Blas.hpp"
#include "MultiDiff.hpp"
#include "SingleDiff.hpp"
#include "cholesky_solve.hpp"
#include "line_search.hpp"


/** Namespace for nonlinear least squares routine */
namespace cpplsq
{

using namespace simd;

struct Verbose {};
struct Silent {};

namespace internal
{
struct IdentityTransform
{
   template<typename REAL>
   const REAL *operator()( const REAL *params ) const
   {
      return params;
   }

   /**
   * Do some initialization here. The MultiDiff::Context will be
   * initialized before this call, so here it is allowed to pre-
   * allocate MultiDiff objects.
   */
   void num_parameters( std::size_t N ) const {}
};


template<typename VERBOSITY>
struct Stream;

template<>
struct Stream<Verbose>
{
   template<typename T>
   std::ostream &operator<< ( const T &str )
   {
      std::cout << str;
      return std::cout;
   }
};

template<>
struct Stream<Silent>
{

   template<typename T>
   const Stream<Silent> &operator<< ( const T &val ) const
   {
      return *this;
   }
};
}

/**
 * \brief Compute parameters such that the sum of squares of the (nonlinear) residuals is minimized.
 *
 * \param tolerance             Value to use for tolerance. If the change in the function value (sum of squared residuals) is smaller than tolerance
 *                              for 15 consecutive iterations or if the max norm of the gradient is smaller than tolerance the algorithm terminates.
 * \param params                On input contains the initial parameters and on output the parameters that minimize the sum of squares of the residuals.
 * \param residuals             An array or vector of residual functors. The input of each functor is a pointer to parameters to use for the computation.
 *                              The type of parameters may is a SingleDiff or MultiDiff type to compute derivatives together with function values and thus
 *                              the functors must be templated to accept different types.
 * \param parameterTransform    An optional functor that may perform a transformation on the parameters. Defaults to identity function i.e. no transformation.
 *                              If a reference is returned a copy will be made so use a pointer if this is unwanted. If this parameter is used the input
 *                              of the residual functors will be the type of the transformed parameters as returned by this functor. The functor must not
 *                              preallocate any MultiDiff objects until its member function num_parameters(N) is called which happens after the MultiDiff
 *                              context is initialized.
 * \tparam VERBOSITY            If set to cpplsq::Verbose then there will be output to stdout in each iteration. If set to cpplsq::Silent then there is no output.
 *                              Default value is cpplsq::Verbose.
 * \tparam MAXITER              Maximum number of iterations that will be performed. Default value is 1000.
 *
 */
template<typename VERBOSITY = Verbose , int MAXITER = 1000, typename REAL, typename Residuals, typename ParameterTransform = internal::IdentityTransform>
void gn_sbfgs_min( REAL tolerance, simd::aligned_vector<REAL> &params, Residuals residuals, ParameterTransform parameterTransform = ParameterTransform() )
{
   internal::Stream<VERBOSITY>() << std::left << std::scientific;
   using std::size_t;
   using std::unique_ptr;
   const size_t N = params.size();
   const size_t CN = simd::next_size<REAL> ( N );
   const size_t M = residuals.size();
   const size_t NxCN = N * CN;

   using SD = SingleDiff<REAL>;
   using MD = MultiDiff<REAL>;
   using MDContext = typename MD::Context;

   using array = simd::aligned_array<REAL>;
   auto new_array = []( size_t n )
   {
      return  simd::alloc_aligned_array<REAL> ( n );
   };

   MDContext ctx( N );
   {
      //move into scope that gets destroyed before the MultiDiff::Context
      ParameterTransform pt = std::move( parameterTransform );
      //now call init function of tranformator
      pt.num_parameters( N );

      unique_ptr<MD[]> ad_params( new MD[N] );
      unique_ptr<SD[]> directed_ad_params( new SD[N] );
      unique_ptr<MD[]> r( new MD[M] );

      array g = new_array( CN );
      array s = new_array( CN );
      array z = new_array( CN );
      array As = new_array( CN );

      //array J_ = new_array(2);
      //auto J = [&](size_t i, size_t j) -> REAL& { return J_[i*CN+j]; };

      array B_ = new_array( NxCN );
      auto B = [&]( size_t i, size_t j ) -> REAL& { return B_[i * CN + j]; };

      array A_ = new_array( NxCN );
      auto A = [&]( size_t i, size_t j ) -> REAL& { return A_[i * CN + j]; };

      aligned_fill( zero<REAL>(), A_.get(), A_.get() + NxCN );
      aligned_fill( zero<REAL>(), g.get(), g.get() + CN );

      REAL normr2 = 0;

      for( size_t i = 0; i < N; ++i )
      {
         ad_params[i].setIndependent( params[i], i );
      }

      aligned_fill( zero<REAL>(), B_.get(), B_.get() + NxCN );

      {
         auto tp = pt( ad_params.get() );

         for( size_t i = 0; i < M; ++i )
         {
            MD residual = residuals[i]( tp );
            normr2 += residual.getValue() * residual.getValue();
            const pack<REAL> rval( residual.getValue() );
            blas::axpy( N, residual.getValue(), residual.getDiffValues(), 1, g.get(), 1 );
            blas::syr( N, 1.0, residual.getDiffValues(), 1, B_.get(), CN );
            r[i] = std::move( residual );
         }
      }

      REAL normr = 1e-4 * std::sqrt( normr2 );

      for( size_t i = 0; i < N; ++i )
      {
         ad_params[i].setIndependent( params[i], i );
         A( i, i ) = normr;
      }

      aligned_transform<1>(
         []( std::array<pack<REAL>, 2> &p )
      {
         p[0] += p[1];
      },
      NxCN, B_.get(), A_.get()
      );

      int small_progress = 0;

      for( int k = 0; k < MAXITER; ++k )
      {

         // copyneg ( CN, s, g );
         aligned_transform( []( const pack<REAL> &g )
         {
            return -g;
         }, s.get(), s.get() + CN, g.get() );
         int pos = cholesky_solve( B_, CN, s, N );

         if( pos )
         {
            //use gradient descent
            aligned_transform( []( const pack<REAL> &g )
            {
               return -g;
            }, s.get(), s.get() + CN, g.get() );
         }

         REAL alpha = 1;
         SD f0;
         f0 = 0.5 * normr2, blas::dot( N, g.get(), 1, s.get(), 1 );
         auto eval_step_size = [&]( REAL a ) -> SD
         {
            for( size_t i = 0; i < N; ++i )
            {
               directed_ad_params[i] = params[i] + a * s[i], s[i];
            }

            auto tp = pt( directed_ad_params.get() );
            SD f = 0;

            for( size_t i = 0; i < M; ++i )
            {
               SD residual = residuals[i]( tp );
               f += residual * residual;

            }

            return f * 0.5;
         };

         bool found_step_size = line_search( f0, eval_step_size, alpha );

         if( found_step_size )
         {
            //scale step by step size alpha

            blas::scal( N, alpha, s.get(), 1 );

            //set params = params + step
            for( size_t i = 0; i < N; ++i )
            {
               REAL pi = params[i] + s[i];
               params[i] = pi;
               ad_params[i].setIndependent( pi, i );
            }

            //evaluate residuals gradient and z = (J1 - J0)^T * r1 * norm(r1)/norm(r0)
            auto tp = pt( ad_params.get() );
            REAL new_normr2 = 0;
            //set B, g and z zero
            {
               const pack<REAL> zp = zero<REAL>();
               aligned_fill( zp, B_.get(), B_.get() + NxCN );
               aligned_fill( zp, g.get(), g.get() + CN );
               aligned_fill( zp, z.get(), z.get() + CN );
            }

            for( size_t i = 0; i < M; ++i )
            {
               MD residual = residuals[i]( tp );
               new_normr2 += residual.getValue() * residual.getValue();

               const pack<REAL> rval( residual.getValue() );
               aligned_transform<2>(
                  [&rval]( std::array<pack<REAL>, 4> &p )
               {
                  p[0] += rval * p[2];
                  p[1] += rval * ( p[2] - p[3] );
               },
               CN,
               g.get(),  z.get(), residual.getDiffValues(), r[i].getDiffValues()
               );

               blas::syr( N, 1.0, residual.getDiffValues(), 1, B_.get(), CN );
               r[i] = std::move( residual );
            }

            blas::scal( N, std::sqrt( new_normr2 / normr2 ), z.get(), 1 );

            REAL delta = 0.5 * ( normr2 - new_normr2 );

            if( delta < tolerance )
               ++small_progress;
            else
               small_progress = 0;

            int imax = blas::iamax( N, g.get(), 1 );

            internal::Stream<VERBOSITY>() << "itr: " <<  std::setw( 6 ) <<  k + 1  << "r: " << std::setw( 14 ) << 0.5 * normr2  <<   "d: "   << std::setw( 14 ) << delta   <<  "g: " << std::setw( 14 )  << std::abs( g[imax] );

            if( small_progress == 15 )
            {
               internal::Stream<VERBOSITY>() << "\nchange in function value was smaller than tolerance for 15 consecutive iterations\n";
               break;
            }

            if( std::abs( g[imax] ) < tolerance )
            {
               internal::Stream<VERBOSITY>() << "\ngradient max norm smaller than tolerance.\n";
               break;
            }

            normr2 = new_normr2;
            //compute next A and B
            REAL zs = blas::dot( N, z.get(), 1, s.get(), 1 );

            if( zs / blas::dot( N, s.get(), 1, s.get(), 1 ) >= 1e-6 )
            {
               internal::Stream<VERBOSITY>() << "H: SBFGS\n";
               //As = A*s
               blas::symv( N, REAL( 1 ), A_.get(), CN, s.get(), 1, REAL( 0 ), As.get(), 1 );
               REAL sAs = blas::dot( N, s.get(), 1, As.get(), 1 );
               blas::syr( N, -1 / sAs, As.get(), 1, A_.get(), CN );
               blas::syr( N, 1 / zs, z.get(), 1, A_.get(), CN );
               aligned_transform<1>(
                  []( std::array<pack<REAL>, 2> &p )
               {
                  p[0] += p[1];
               },
               NxCN, B_.get(), A_.get()
               );
            }
            else
            {
               internal::Stream<VERBOSITY>() << "H: GN\n";
               REAL normr = std::sqrt( normr2 );

               for( size_t i = 0; i < N; ++i )
                  B( i, i ) += normr;
            }

         }
         else
         {
            internal::Stream<VERBOSITY>() << "no step satisfying the weak wolfe conditions was found\n";
            break;
         }
      }

      internal::Stream<VERBOSITY>() << std::resetiosflags( std::ios::floatfield | std::ios::adjustfield );

   } //ctx gets destroyed
} //end of gn_sbfgs_min

} //clsq

#endif
