from __future__ import annotations

from alphagrid.agents.graph_rag import EnergyKnowledgeGraph, query_graph_context


def test_knowledge_graph_bootstrap():
    kg = EnergyKnowledgeGraph()
    assert "DE_LU" in kg.graph
    assert "TTF_Gas" in kg.graph
    assert kg.graph.has_edge("TTF_Gas", "DE_LU")


def test_dynamic_relationship_addition():
    kg = EnergyKnowledgeGraph()
    kg.add_relationship("Unplanned_Outage_Nord1", "DE_LU", "REDUCES_CAPACITY")
    assert "Unplanned_Outage_Nord1" in kg.graph
    assert kg.graph.has_edge("Unplanned_Outage_Nord1", "DE_LU")

    context = kg.get_summary_context("DE_LU")
    assert "Unplanned_Outage_Nord1 --[REDUCES_CAPACITY]--> DE_LU" in context


def test_query_graph_context_helper():
    ctx = query_graph_context("DE_LU")
    assert "GraphRAG Knowledge Context for DE_LU:" in ctx
    assert "TTF_Gas --[DRIVES_MARGINAL_COST]--> DE_LU" in ctx
