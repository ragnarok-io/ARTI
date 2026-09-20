"""Historical Federal/TensorView program names for artifact inspection only."""

from typing import ClassVar, Mapping

from ..federal_tensor_view import FederatedProgram, RoutedProgram
from ..terminal_abi import TerminalOutputABI


class TensorViewFormulaProgram(RoutedProgram):
    """Historical local-program identity retained for artifact inspection."""

    _component_reference: ClassVar[str] = "arti/tensor-view-formula-program@2"


class FederalRecallV3(FederatedProgram):
    """Historical federation identity retained for artifact inspection."""

    _component_reference: ClassVar[str] = "arti/federal-recall@3"

    def __init__(
        self,
        programs: Mapping[str, RoutedProgram],
        *,
        terminal_abi: TerminalOutputABI,
        root_bank_ids: tuple[str, ...],
        max_levels: int = 8,
        max_k: int = FederatedProgram.recommended_breadth,
        winner_policy: str = "hard_one_winner",
    ) -> None:
        super().__init__(
            programs,
            terminal_abi=terminal_abi,
            root_program_ids=root_bank_ids,
            max_levels=max_levels,
            max_k=max_k,
            winner_policy=winner_policy,
        )

    def contract_config(self) -> dict[str, object]:
        return {
            "terminal_abi": self.terminal_abi.to_dict(),
            "root_bank_ids": list(self.root_program_ids),
            "bank_signatures": {
                program_id: self.programs[program_id].signature.to_dict()
                for program_id in sorted(self.programs)
            },
            "max_levels": self.max_levels,
            "max_k": self.max_k,
            "winner_policy": self.winner_policy,
        }

__all__ = ["FederalRecallV3", "TensorViewFormulaProgram"]
