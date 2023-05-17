from typing import List, Optional, Tuple
import theseus as th
import torch

from covpred.math.common import sphere_to_carthesian
from covpred.math.pose_optimization import fast_denominator, fast_denominator_jac
from covpred.model.optimization.theseus.nec import NECCost


class PNECCost(NECCost):
    def __init__(
        self,
        cost_weight: th.CostWeight,
        rotations: th.SO3,
        translations: th.Variable,
        bvs_0_hat: th.Variable,
        bvs_1: th.Variable,
        covariances_0: th.Variable,
        covariances_1: th.Variable,
        regularization: th.Variable,
        scaling: th.Variable,
        opt_R: bool = True,
        opt_t: bool = True,
        name: Optional[str] = None,
    ):
        super().__init__(
            cost_weight=cost_weight,
            rotations=rotations,
            translations=translations,
            bvs_0_hat=bvs_0_hat,
            bvs_1=bvs_1,
            scaling=scaling,
            opt_R=opt_R,
            opt_t=opt_t,
            name=name,
        )
        self.covariances_0 = covariances_0
        self.covariances_1 = covariances_1
        self.regularization = regularization
        self.register_aux_vars(["covariances_0", "covariances_1", "regularization"])

    def _denominator(self) -> torch.Tensor:
        carthesian_translation, _ = sphere_to_carthesian(self.translations.tensor)
        return fast_denominator(
            carthesian_translation,
            self.rotations.tensor,
            self.bvs_0_hat.tensor,
            self.covariances_0.tensor,
            self.bvs_1.tensor,
            self.covariances_1.tensor,
            self.regularization.tensor,
        )

    def error(self) -> torch.Tensor:
        return self._numerator() / self._denominator()

    def _denominator_jac(self) -> Tuple[torch.Tensor, torch.Tensor]:
        carthesian_translation, carthesian_jac = sphere_to_carthesian(self.translations.tensor, jacobian=True)
        jac_rot, jac_t = fast_denominator_jac(
            carthesian_translation,
            self.rotations.tensor,
            self.bvs_0_hat.tensor,
            self.covariances_0.tensor,
            self.bvs_1.tensor,
            self.covariances_1.tensor,
            self.regularization.tensor,
        )

        return jac_rot, torch.einsum("B...i,Bij->B...j", jac_t, carthesian_jac)

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        numerator, denominator = self._numerator(), self._denominator()
        num_jac_rot, num_jac_t = self._numerator_jac()
        den_jac_rot, den_jac_t = self._denominator_jac()
        jacs = []
        if self.opt_R:
            jacs.append(
                torch.divide(num_jac_rot, denominator[..., None])
                - torch.divide(
                    torch.mul(numerator[..., None], den_jac_rot),
                    torch.square(denominator)[..., None],
                )
            )
        if self.opt_t:
            jacs.append(
                torch.divide(num_jac_t, denominator[..., None])
                - torch.divide(
                    torch.mul(numerator[..., None], den_jac_t),
                    torch.square(denominator)[..., None],
                )
            )
        return jacs, numerator / denominator

    def _copy_impl(self, new_name: Optional[str] = None) -> "PNECCost":
        return PNECCost(  # type: ignore
            self.weight.copy(),
            self.rotations.copy(),
            self.translations.copy(),
            self.bvs_0_hat.copy(),
            self.bvs_1.copy(),
            self.covariances_0.copy(),
            self.covariances_1.copy(),
            self.regularization.copy(),
            self.scaling.copy(),
            self.opt_R,
            self.opt_t,
            name=new_name,
        )
