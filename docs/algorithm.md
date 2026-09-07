# Algorithm and numerical policies

For a real-valued signal `y(x)` sampled at uniform interval `delta`, FDSM applies
the following pipeline.

1. Optionally smooth the signal with a Savitzky-Golay filter.
2. Scale the full smoothed signal using the mean and population standard deviation
   of the central retained region.
3. Reflect-pad both endpoints.
4. For every requested derivative order `alpha`, calculate
   `(i 2 pi f)^alpha FFT(y)` on the principal complex branch.
5. Apply the inverse FFT, remove padding, and retain its real component.
6. Crop a fixed proportion from both signal-axis endpoints.
7. Independently z-score every derivative-order row.

The final array has shape `(number of orders, retained signal length)`. Columns
remain aligned to the same input coordinate across all derivative orders.

## Sampling axis

FFT differentiation requires uniform sampling. The library accepts ascending and
descending uniform axes and uses the median observed spacing as `delta`. It checks
the maximum spacing deviation against

`uniformity_atol + uniformity_rtol * abs(delta)`.

A truly nonuniform axis raises by default. Mapping a nonuniform axis to `[0, 1]`
does not make it uniformly sampled, so the library does not use axis normalization
as a substitute for resampling.

An affine change of a uniform axis changes derivative magnitude by a constant
factor for each order. With the default independent row normalization, this factor
is largely removed, while the physical coordinate and sampling interval remain
available in metadata.

## Order ladder

The convenience API includes both endpoints and therefore requires
`(max_order - min_order) / order_step` to be an integer within numerical
tolerance. Arbitrary nonuniform order sets can be supplied through `orders=`.

Orders must be finite and nonnegative. Negative orders introduce a singular
zero-frequency multiplier and are outside this implementation.

## Nyquist policy

The principal-branch multiplier is conjugate symmetric on paired frequency bins.
For an even transform length, however, the Nyquist bin is unpaired. At a
noninteger order its multiplier can be complex, leaving a small imaginary
component in the inverse transform.

- `retain`: retain the coefficient, calculate the complex inverse FFT, and return
  the real component.
- `zero_unpaired`: set that coefficient to zero when its multiplier has a
  meaningful imaginary component.

The selected policy is recorded in result metadata.
