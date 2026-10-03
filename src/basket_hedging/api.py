"""The compact public entry point mirrors the interview pseudocode."""

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

from .contracts import InputError, Instrument, Opening
from .engine import Engine, Policy, PolicyView, Result
from .market import Event, Pricer, Rules
from .risk import MarginModel, UnnettedLiabilityStressMargin


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float
    hedge_instruments: tuple[Instrument, ...]
    rules: Rules
    pricer: Pricer | None = None
    margin_model: MarginModel | None = None
    liability_quantity: float = -1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "hedge_instruments", tuple(self.hedge_instruments))
        ids = [instrument.id for instrument in self.hedge_instruments]
        if len(ids) != len(set(ids)) or any(instrument.is_basket for instrument in self.hedge_instruments):
            raise InputError("registered hedge instruments must be unique and exclude the basket")


def backtest_engine(basket_option: Instrument, market_data: Iterable[Event],
                    hedging_strategy: Policy | Callable[[PolicyView], Mapping[str, float] | None],
                    config: BacktestConfig) -> Result:
    """Originate once, then replay. Engine + Opening('IMPORT', ...) loads an existing book.

    The inception record must supply the actual sale price. A model value is
    never implicitly substituted for a premium or an executable hedge quote.
    """
    if not basket_option.is_basket:
        raise InputError("basket_option must be an arithmetic basket call or put")
    instruments = {instrument.id: instrument for instrument in config.hedge_instruments}
    if basket_option.id in instruments:
        raise InputError("basket ID conflicts with a hedge instrument")
    instruments[basket_option.id] = basket_option
    opening = Opening("ORIGINATE", initial_capital=config.initial_capital,
                      liability_id=basket_option.id, liability_quantity=config.liability_quantity)
    margin = config.margin_model if config.margin_model is not None else UnnettedLiabilityStressMargin(
        expected_scenarios=config.rules.expected_stress_scenarios)
    return Engine(instruments, opening, config.rules, margin, config.pricer).run(market_data, hedging_strategy)
