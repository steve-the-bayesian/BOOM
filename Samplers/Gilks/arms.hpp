// code taken from Wally Gilks' web site
/* header file for arms function */

namespace GilksArms {
  enum GilksErrorCode {
    Success = 0,
    TooFewInitialPoints,
    TooManyInitialPoints,
    InitialPointsDoNotSatisfyBounds,
    DataNotOrdered,
    NegativeConvexityParameter,
    InsufficientSpace,
    EnvelopeViolation,
    CentileOutOfRange,
    PreviousIterateOutOfRange
  };

  GilksErrorCode arms_simple(BOOM::RNG &, int ninit, double *xl, double *xr,
                             double (*myfunc)(double x, void *mydata), void *mydata,
                             int dometrop, double *xprev, double *xsamp);

  GilksErrorCode arms(BOOM::RNG &, double *xinit, int ninit, double *xl, double *xr,
                      double (*myfunc)(double x, void *mydata), void *mydata,
                      double *convex, int npoint, int dometrop, double *xprev,
                      double *xsamp, int nsamp, double *qcent, double *xcent, int ncent,
                      int *neval);

  double expshift(double y, double y0);

}  // namespace GilksArms
