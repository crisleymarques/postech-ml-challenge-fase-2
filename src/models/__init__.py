"""Modelos de recomendação do projeto."""

from .baselines import BaseRecommender, ItemKNNRecommender, PopularityRecommender

__all__ = [
    "BaseRecommender",
    "ItemKNNRecommender",
    "PopularityRecommender",
]
