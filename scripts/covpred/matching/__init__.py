from enum import Enum

from thirdparty.SuperGluePretrainedNetwork.models.matching import Matching

from covpred.matching.base import MatchingFunction
from covpred.matching.superglue import SuperGlue
from covpred.matching.klt import KLT
from covpred.matching.orb import ORB


class MatchingAlgorithm(Enum):
    SUPERGLUE = 1
    KLT = 2
    ORB = 3
