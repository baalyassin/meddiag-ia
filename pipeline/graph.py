"""
Graphe LangGraph — Pipeline de Raisonnement Diagnostique
4 nœuds en cascade : PI → IA₁ → IA₂ → IA₃
"""
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from pipeline.nodes import (
    initialization_node,
    generation_node,
    evaluation_node,
    predictive_node,
)


class PipelineState(TypedDict):
    """État partagé entre les nœuds du pipeline"""
    structured_data: dict                    # Données patient structurées (Phase 2 output)
    step_outputs: dict[str, str]             # Sorties accumulées : step_1, step_2, step_3, step_4
    current_step: int                        # Étape courante (0-4)
    diagnostic: Optional[dict]              # Résultat final du diagnostic


def build_pipeline_graph() -> StateGraph:
    """
    Construit et compile le graphe LangGraph du pipeline diagnostique.

    Architecture :
    ┌──────────────────────────────────────────────────┐
    │                   Pipeline IA                     │
    │                                                   │
    │  [init_node] → [gen_node] → [eval_node] → [pred] │
    │      PI           IA₁          IA₂          IA₃  │
    │                                             ↓     │
    │                                           [END]   │
    └──────────────────────────────────────────────────┘
    """
    graph = StateGraph(PipelineState)

    # ① Nœud d'initialisation — contextualise le cas clinique
    graph.add_node("initialization", initialization_node)

    # ② Nœud de génération — premier raisonnement diagnostique
    graph.add_node("generation", generation_node)

    # ③ Nœud d'évaluation — évalue et optimise le prompt
    graph.add_node("evaluation", evaluation_node)

    # ④ Nœud prédictif — génère les hypothèses diagnostiques finales
    graph.add_node("predictive", predictive_node)

    # Flux séquentiel
    graph.set_entry_point("initialization")
    graph.add_edge("initialization", "generation")
    graph.add_edge("generation", "evaluation")
    graph.add_edge("evaluation", "predictive")
    graph.add_edge("predictive", END)

    return graph.compile()
