# https://github.com/willisma/SiT/
import torch as th
import enum

from . import paths
from .solvers import ode


def mean_flat(x, mask):
    return th.sum(x * mask, dim=list(range(1, len(x.size())))) / th.sum(mask, dim=list(range(1, len(x.size()))))


class ModelType(enum.Enum):
    NOISE = enum.auto()
    SCORE = enum.auto()
    VELOCITY = enum.auto()


class PathType(enum.Enum):
    LINEAR = enum.auto()
    GVP = enum.auto()
    VP = enum.auto()


class WeightType(enum.Enum):
    NONE = enum.auto()
    VELOCITY = enum.auto()
    LIKELIHOOD = enum.auto()


class Transport:

    def __init__(self, *, args, model_type, path_type, loss_type, train_eps, sample_eps):
        path_options = {
            PathType.LINEAR: paths.ICPlan,
            PathType.GVP: paths.GVPCPlan,
            PathType.VP: paths.VPCPlan,
        }
        self.args = args
        self.loss_type = loss_type
        self.model_type = model_type
        self.path_sampler = path_options[path_type]()
        self.train_eps = train_eps
        self.sample_eps = sample_eps

    def check_interval(self, train_eps, sample_eps, *, diffusion_form="SBDM", sde=False,
                        reverse=False, eval=False, last_step_size=0.0):
        t0 = 0
        t1 = 1
        eps = train_eps if not eval else sample_eps
        if (type(self.path_sampler) in [paths.VPCPlan]):
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size
        elif (type(self.path_sampler) in [paths.ICPlan, paths.GVPCPlan]) \
                and (self.model_type != ModelType.VELOCITY or sde):
            t0 = eps if (diffusion_form == "SBDM" and sde) or self.model_type != ModelType.VELOCITY else 0
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size
        if reverse:
            t0, t1 = 1 - t0, 1 - t1
        return t0, t1

    def sample(self, x1):
        x0 = th.randn_like(x1)
        t0, t1 = self.check_interval(self.train_eps, self.sample_eps)
        t = th.rand((x1.shape[0],)) * (t1 - t0) + t0
        t = t.to(x1)
        return t, x0, x1

    def training_losses(self, model, x1, aatype1=None, mask=None, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}

        t, x0, x1 = self.sample(x1)
        t, xt, ut = self.path_sampler.plan(t, x0, x1)

        model_output = model(xt, t, **model_kwargs)
        B, *_, C = xt.shape
        assert model_output.size() == (B, *xt.size()[1:-1], C)

        terms = {}
        terms['t'] = t
        terms['pred'] = model_output
        if self.model_type == ModelType.VELOCITY:
            terms['loss'] = mean_flat(((model_output - ut) ** 2), mask)
        else:
            _, drift_var = self.path_sampler.compute_drift(xt, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(paths.expand_t_like_x(t, xt))
            if self.loss_type in [WeightType.VELOCITY]:
                weight = (drift_var / sigma_t) ** 2
            elif self.loss_type in [WeightType.LIKELIHOOD]:
                weight = drift_var / (sigma_t ** 2)
            elif self.loss_type in [WeightType.NONE]:
                weight = 1
            else:
                raise NotImplementedError()

            if self.model_type == ModelType.NOISE:
                terms['loss'] = mean_flat(weight * ((model_output - x0) ** 2), mask)
            else:
                terms['loss'] = mean_flat(weight * ((model_output * sigma_t + x0) ** 2), mask)

        return terms

    def get_drift(self):
        def score_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            model_output = model(x, t, **model_kwargs)
            return (-drift_mean + drift_var * model_output)

        def noise_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(paths.expand_t_like_x(t, x))
            model_output = model(x, t, **model_kwargs)
            score = model_output / -sigma_t
            return (-drift_mean + drift_var * score)

        def velocity_ode(x, t, model, **model_kwargs):
            model_output = model(x, t, **model_kwargs)
            return model_output

        if self.model_type == ModelType.NOISE:
            drift_fn = noise_ode
        elif self.model_type == ModelType.SCORE:
            drift_fn = score_ode
        else:
            drift_fn = velocity_ode

        def body_fn(x, t, model, **model_kwargs):
            model_output = drift_fn(x, t, model, **model_kwargs)
            assert model_output.shape == x.shape, "Output shape from ODE solver must match input shape"
            return model_output

        return body_fn

class Sampler:
    """Sampler for the transport model."""

    def __init__(self, transport):
        self.transport = transport
        self.drift = self.transport.get_drift()

    def sample_ode(self, *, sampling_method="dopri5", num_steps=50, atol=1e-6, rtol=1e-3, reverse=False):
        if reverse:
            drift = lambda x, t, model, **kwargs: self.drift(x, th.ones_like(t) * (1 - t), model, **kwargs)
        else:
            drift = self.drift

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps, self.transport.sample_eps,
            sde=False, eval=True, reverse=reverse, last_step_size=0.0)

        _ode = ode(drift=drift, t0=t0, t1=t1, sampler_type=sampling_method,
                   num_steps=num_steps, atol=atol, rtol=rtol)
        return _ode.sample

def create_transport(args, path_type='Linear', prediction="velocity",
                     loss_weight=None, train_eps=None, sample_eps=None):
    """Factory function to create a Transport object."""
    if prediction == "noise":
        model_type = ModelType.NOISE
    elif prediction == "score":
        model_type = ModelType.SCORE
    else:
        model_type = ModelType.VELOCITY

    if loss_weight == "velocity":
        loss_type = WeightType.VELOCITY
    elif loss_weight == "likelihood":
        loss_type = WeightType.LIKELIHOOD
    else:
        loss_type = WeightType.NONE

    path_choice = {"Linear": PathType.LINEAR, "GVP": PathType.GVP, "VP": PathType.VP}
    path_type = path_choice[path_type]

    if (path_type in [PathType.VP]):
        train_eps = 1e-5 if train_eps is None else train_eps
        sample_eps = 1e-3 if sample_eps is None else sample_eps
    elif (path_type in [PathType.GVP, PathType.LINEAR] and model_type != ModelType.VELOCITY):
        train_eps = 1e-3 if train_eps is None else train_eps
        sample_eps = 1e-3 if sample_eps is None else sample_eps
    else:
        train_eps = 0
        sample_eps = 0

    return Transport(args=args, model_type=model_type, path_type=path_type,
                     loss_type=loss_type, train_eps=train_eps, sample_eps=sample_eps)
