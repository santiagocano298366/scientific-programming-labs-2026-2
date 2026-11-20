"""Retention curve and numerical methods."""
import numpy as np
from scipy.optimize import curve_fit


class SoilRetentionCurve:
    """A measured retention curve."""

    def __init__(self, layer_id, psi, theta):
        """Save the data of the curve."""
        order = np.argsort(psi)
        self.layer_id = layer_id
        self.psi = np.array(psi)[order]
        self.theta = np.array(theta)[order]

    def theta_at(self, psi_values):
        """Interpolate theta at psi."""
        return np.interp(psi_values, self.psi, self.theta)

    def theta_at_log(self, log_psi):
        """Interpolate theta at log psi."""
        return self.theta_at(10 ** np.asarray(log_psi))


class Integrator:
    """Base class of the integration rules."""

    def __init__(self, a, b, n):
        """Build the points and weights."""
        self.a, self.b, self.n = a, b, n
        self.points, self.weights = self.construct_method()

    def construct_method(self):
        """Points and weights of the rule."""
        pass

    def integrate(self, f):
        """Integrate the function f."""
        return self.weights @ f(self.points)


class Trapezoidal(Integrator):
    """Trapezoidal rule."""

    def construct_method(self):
        """Points and weights of the rule."""
        h = (self.b - self.a) / (self.n - 1)
        w = h * np.ones(self.n)
        w[0] = w[-1] = h / 2
        x = np.linspace(self.a, self.b, self.n)
        return x, w


class Derivative:
    """Base class of the numerical derivatives."""

    def __init__(self, func, h=1e-4):
        """Save the function and the step h."""
        self.func = func
        self.h = h


class ForwardDiff(Derivative):
    """Forward difference."""

    def __call__(self, x):
        """Compute the derivative at x."""
        return (self.func(x + self.h) - self.func(x)) / self.h


class CentralDiff(Derivative):
    """Central difference."""

    def __call__(self, x):
        """Compute the derivative at x."""
        return (self.func(x + self.h) - self.func(x - self.h)) / (2 * self.h)


def van_genuchten(psi, alpha, n, theta_r, theta_s):
    """Van Genuchten model."""
    m = 1 - 1 / n
    return theta_r + (theta_s - theta_r) / (1 + (alpha * psi) ** n) ** m


class VanGenuchtenModel:
    """Van Genuchten fit of one curve."""

    def __init__(self):
        """Start without parameters."""
        self.params = None

    def fit(self, curve):
        """Fit the four parameters."""
        p0 = [1.0, 1.5, min(curve.theta.min(), 0.4), max(curve.theta.max(), 0.3)]
        lower = [0.001, 1.01, 0.0, 0.2]
        upper = [1000, 10.0, 0.5, 1.0]
        self.params, _ = curve_fit(van_genuchten, curve.psi, curve.theta, p0=p0, bounds=(lower, upper), maxfev=10000)
        return self

    def __call__(self, psi):
        """Evaluate the fitted curve."""
        return van_genuchten(psi, *self.params)

    def rmse(self, curve):
        """Root mean squared error of the fit."""
        return np.sqrt(np.mean((curve.theta - self(curve.psi)) ** 2))
