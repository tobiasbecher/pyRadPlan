from ._base_tradeoff_strategies import TradeoffStrategyBase

class SinglePlan(TradeoffStrategyBase):
    name = "Single Plan Tradeoff Strategy"
    short_name = "single"
    ScalarizationStrategy = "WeightedSum"
