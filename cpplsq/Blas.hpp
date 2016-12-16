#ifndef _CPPLSQ_BLAS_HPP_
#define _CPPLSQ_BLAS_HPP_

#include <cblas.h>
namespace cpplsq
{
/**
 * Wrapper for blas functions with float and double overloads
 * using same storage orders for each call.
 */
template<CBLAS_UPLO Uplo, CBLAS_ORDER Order>
struct Blas
{
   static void axpy( const int N, const double alpha, const double *X, const int INCX, double *Y, const int INCY )
   {
      cblas_daxpy( N, alpha, X, INCX, Y, INCY );
   }

   static void axpy( const int N, const float alpha, const float *X, const int INCX, float *Y, const int INCY )
   {
      cblas_saxpy( N, alpha, X, INCX, Y, INCY );
   }

   static void syr( const int N, const double alpha, const double *X,
                    const int incX, double *A, const int lda )
   {
      cblas_dsyr( Order, Uplo, N, alpha, X, incX, A, lda );
   }


   static void syr( const int N, const float alpha, const float *X,
                    const int incX, float *A, const int lda )
   {
      cblas_ssyr( Order, Uplo, N, alpha, X, incX, A, lda );
   }

   static void symv( const int N, const double alpha, const double *A,
                     const int lda, const double *X, const int incX,
                     const double beta, double *Y, const int incY )
   {
      cblas_dsymv( Order, Uplo, N, alpha, A, lda, X, incX, beta, Y, incY );
   }

   static void symv( const int N, const float alpha, const float *A,
                     const int lda, const float *X, const int incX,
                     const float beta, float *Y, const int incY )
   {
      cblas_ssymv( Order, Uplo, N, alpha, A, lda, X, incX, beta, Y, incY );
   }

   static double dot( const int N, const double *X, const int incX,
                      const double *Y, const int incY )
   {
      return cblas_ddot( N, X, incX, Y, incY );
   }

   static float dot( const int N, const float *X, const int incX,
                     const float *Y, const int incY )
   {
      return cblas_sdot( N, X, incX, Y, incY );
   }

   static void gemv( const CBLAS_TRANSPOSE TransA, const int M, const int N,
                     const double alpha, const double *A, const int lda,
                     const double *X, const int incX, const double beta,
                     double *Y, const int incY )
   {
      cblas_dgemv( Order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY );
   }


   static void gemv( const CBLAS_TRANSPOSE TransA, const int M, const int N,
                     const float alpha, const float *A, const int lda,
                     const float *X, const int incX, const float beta,
                     float *Y, const int incY )
   {
      cblas_sgemv( Order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY );
   }

   static void scal( const int N, const double alpha, double *X, const int incX )
   {
      cblas_dscal( N, alpha, X, incX );
   }

   static void scal( const int N, const float alpha, float *X, const int incX )
   {
      cblas_sscal( N, alpha, X, incX );
   }

   static void syrk( const CBLAS_TRANSPOSE Trans, const int N, const int K,
                     const double alpha, const double *A, const int lda,
                     const double beta, double *C, const int ldc )
   {
      cblas_dsyrk( Order, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc );
   }

   static void syrk( const CBLAS_TRANSPOSE Trans, const int N, const int K,
                     const float alpha, const float *A, const int lda,
                     const float beta, float *C, const int ldc )
   {
      cblas_ssyrk( Order, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc );
   }

   static void symm( const CBLAS_SIDE Side, const int M, const int N,
                     const double alpha, const double *A, const int lda,
                     const double *B, const int ldb, const double beta,
                     double *C, const int ldc )
   {
      cblas_dsymm( Order, Side, Uplo, M, N, alpha, A, lda, B, ldb, beta, C, ldc );
   }

   static void symm( const CBLAS_SIDE Side, const int M, const int N,
                     const float alpha, const float *A, const int lda,
                     const float *B, const int ldb, const float beta,
                     float *C, const int ldc )
   {
      cblas_ssymm( Order, Side, Uplo, M, N, alpha, A, lda, B, ldb, beta, C, ldc );
   }

   static void trsv( const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                     const int N, const double *A, const int lda, double *X,
                     const int incX )
   {
      cblas_dtrsv( Order, Uplo, TransA, Diag, N, A, lda, X, incX );
   }


   static void trsv( const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                     const int N, const float *A, const int lda, float *X,
                     const int incX )
   {
      cblas_strsv( Order, Uplo, TransA, Diag, N, A, lda, X, incX );

   }

   static void trmm( const CBLAS_SIDE Side, const CBLAS_TRANSPOSE TransA,
                     const CBLAS_DIAG Diag, const int M, const int N,
                     const double alpha, const double *A, const int lda,
                     double *B, const int ldb )
   {
      cblas_dtrmm( Order, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb );
   }

   static void trmm( const CBLAS_SIDE Side, const CBLAS_TRANSPOSE TransA,
                     const CBLAS_DIAG Diag, const int M, const int N,
                     const float alpha, const float *A, const int lda,
                     float *B, const int ldb )
   {
      cblas_strmm( Order, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb );
   }

   static double nrm2( const int N, const double *X, const int incX )
   {
      return cblas_dnrm2( N, X, incX );
   }

   static float nrm2( const int N, const float *X, const int incX )
   {
      return cblas_snrm2( N, X, incX );
   }

   static double iamax( const int N, const double *X, const int incX )
   {
      return cblas_idamax( N, X, incX );
   }

   static float iamax( const int N, const float *X, const int incX )
   {
      return cblas_isamax( N, X, incX );
   }

   static double asum( const int N, const double *X, const int incX )
   {
      return cblas_dasum( N, X, incX );
   }

   static float asum( const int N, const float *X, const int incX )
   {
      return cblas_sasum( N, X, incX );
   }

};

/**
 * store lower parts of symmertric matrices and use rowmajor storage order
 * by default
 */
using blas = Blas<CblasLower, CblasRowMajor>;

} //cpplsq

#endif
