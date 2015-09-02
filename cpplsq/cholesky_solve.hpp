#ifndef _CPPLSQ_CHOLESKY_SOLVE_HPP_
#define _CPPLSQ_CHOLESKY_SOLVE_HPP_

#include <cstddef>
#include <simd/alloc.hpp>
#include <simd/pack.hpp>
#include <cmath>
#include "Blas.hpp"

namespace cpplsq
{

/**
 * \brief Solve Ax = b for symmetric positive definite matrix A using cholesky decomposition.
 *
 * \param A_         On input a symmetric positive definite matrix.
 * \param LDA        leading dimension of A_ must be greater or equal to N.
 * \param b          On input the right hand side of linear system on output the solution.
 * \param N          Size of A and b, i.e. A is a NxN matrix and b is a vector of size N.
 *
 * \return           If matrix is symmetric positive definite then returns 0 else returns the index
 *                   of the diagonal entry k for which the submatrix A(k:N,k:N) was not symmetric
 *                   positive definite.
 */
template<typename REAL>
int cholesky_solve( const simd::aligned_array<REAL> &A_, std::size_t LDA, const simd::aligned_array<REAL> &b, std::size_t N )
{
   using namespace simd;
   using std::size_t;
   using std::sqrt;

   auto A = [LDA, &A_]( size_t i, size_t j )-> REAL &
   {
      return A_[i * LDA + j];
   };

   //compute cholesky decomposition column by column
   //first column is only root and scaling
   REAL v = A( 0, 0 );

   if( v < 0 )
      return 1;

   A( 0, 0 ) = sqrt( v );
   blas::scal( N - 1, 1 / A( 0, 0 ), &A( 1, 0 ), LDA );

   for( size_t k = 1; k < N; ++k )
   {
      //after first column call gemv to substract the sums of the elements in front of the diagnoal element A(k,k) from the kth column of A
      blas::gemv( CblasNoTrans, N - k, k, REAL( -1 ), &A( k, 0 ), LDA, &A( k, 0 ), 1, REAL( 1 ), &A( k, k ), LDA );
      //check if diagonal entry now is smaller than zero and return position of negative entry if so
      // to indicate that the input matrix was not positive definite in the submatrix starting at A(k,k)
      v = A( k, k );

      if( v < 0 )
         return k + 1;

      //take the square root of A(k,k)
      A( k, k ) = sqrt( v );
      //divide column below A(k,k) by the value of A(k,k)
      blas::scal( N - k - 1, 1 / A( k, k ), &A( k + 1, k ), LDA );
   }

   //backward substitution
   blas::trsv( CblasNoTrans, CblasNonUnit, N, A_.get(), LDA, b.get(), 1 );
   //forward substitution
   blas::trsv( CblasTrans, CblasNonUnit, N, A_.get(), LDA, b.get(), 1 );
   return 0;
}

} //cpplsq


#endif
