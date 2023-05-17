from typing import List, Optional, Tuple
import theseus as th
import torch

from covpred.math.common import sphere_to_carthesian
from covpred.math.pose_optimization import numerator, numerator_jac


class NECCost(th.CostFunction):
    def __init__(
        self,
        cost_weight: th.CostWeight,
        rotations: th.SO3,
        translations: th.Variable,
        bvs_0_hat: th.Variable,
        bvs_1: th.Variable,
        scaling: th.Variable,
        opt_R: bool = True,
        opt_t: bool = True,
        name: Optional[str] = None,
    ):
        super().__init__(cost_weight=cost_weight, name=name)
        self.rotations = rotations
        self.translations = translations
        self.bvs_0_hat = bvs_0_hat
        self.bvs_1 = bvs_1
        self.scaling = scaling
        self.opt_R = opt_R
        self.opt_t = opt_t

        optim_vars = []
        aux_vars = []
        if self.opt_R:
            optim_vars.append("rotations")
        else:
            aux_vars.append("rotations")
        if self.opt_t:
            optim_vars.append("translations")
        else:
            aux_vars.append("translations")

        if len(optim_vars) == 0:
            print(f"WARNING! No optimization variable chosen, there will be no optmization.")
        aux_vars.extend(["scaling", "bvs_0_hat", "bvs_1"])

        self.register_optim_vars(optim_vars)
        self.register_aux_vars(aux_vars)

    def _numerator(self) -> torch.Tensor:
        carthesian_translation, _ = sphere_to_carthesian(self.translations.tensor)
        return self.scaling.tensor * numerator(
            carthesian_translation,
            self.bvs_0_hat.tensor,
            self.rotations.tensor,
            self.bvs_1.tensor,
        )

    def error(self) -> torch.Tensor:
        return self._numerator()

    def _numerator_jac(self) -> Tuple[torch.Tensor, torch.Tensor]:
        carthesian_translation, carthesian_jac = sphere_to_carthesian(self.translations.tensor, jacobian=True)
        jac_rot, jac_t = numerator_jac(
            carthesian_translation, self.bvs_0_hat.tensor, self.rotations.tensor, self.bvs_1.tensor
        )
        return self.scaling.tensor[:, None] * jac_rot, self.scaling.tensor[:, None] * torch.einsum(
            "B...i,Bij->B...j", jac_t, carthesian_jac
        )

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        num = self._numerator()
        jac_rot, jac_t = self._numerator_jac()
        jacs = []
        if self.opt_R:
            jacs.append(jac_rot)
        if self.opt_t:
            jacs.append(jac_t)
        return jacs, num

    def dim(self) -> int:
        return self.bvs_1.shape[-2]

    def _copy_impl(self, new_name: Optional[str] = None) -> "NECCost":
        return NECCost(  # type: ignore
            self.weight.copy(),
            self.rotations.copy(),
            self.translations.copy(),
            self.bvs_0_hat.copy(),
            self.bvs_1.copy(),
            self.scaling.copy(),
            self.opt_R,
            self.opt_t,
            name=new_name,
        )
