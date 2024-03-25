# [`forwardEuler`](./forwardEuler.m)

This is an implementation of a forward Euler method for `dlarray` that behaves similar to `dlode45`.

## API 

* `y = forwardEuler(net,tspan,y0)` solves the ODE defined by `dydt = net.predict(y)` or `dydt = net.predict(t,y)` with initial value `y0` where `net` is a `dlnetwork` with 1 or 2 inputs. 
* The solution `y` is returned at times `tspan(2:end)`.
* The solve uses a timestep `dt = min(diff(tspan))` by default.

## Technical Details

This function performs two steps:

1. Perform a standard Euler solve to find `y` at `tspan(1):dt:tspan(2)`.
2. Use `dlarray.interp1` to interpolate `y` to the times `tspan`.

In particular `tspan` need not be regularly spaced time steps.

## Uses

* An Euler solve is an affine transformation from one timeslice to the next so provides a convex update function.
* An Euler solve is much faster than an 4th order adaptive solve as in `dlode45` but may be less accurate.

## Notes

* This implementation uses linear `interp1` which may not be optimal.