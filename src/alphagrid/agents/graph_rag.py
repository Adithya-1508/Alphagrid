from __future__ import annotations

import networkx as nx


class EnergyKnowledgeGraph:
    """
    Knowledge Graph RAG (GraphRAG) store for European Energy Grid entities & relationships.
    """

    def __init__(self) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()
        self._bootstrap_default_grid_knowledge()

    def _bootstrap_default_grid_knowledge(self) -> None:
        """Seed core European market grid connections."""
        # Grid Zones
        self.graph.add_node("DE_LU", type="GridZone", label="Germany/Luxembourg")
        self.graph.add_node("FR", type="GridZone", label="France")
        self.graph.add_node("NL", type="GridZone", label="Netherlands")
        self.graph.add_node("DK_1", type="GridZone", label="Denmark West")

        # Fuel & Commodity Drivers
        self.graph.add_node("TTF_Gas", type="Commodity", label="TTF Natural Gas")
        self.graph.add_node("EU_ETS_Carbon", type="Commodity", label="EU ETS Carbon Credits")
        self.graph.add_node("Wind_Power", type="Renewable", label="Wind Power Generation")
        self.graph.add_node("Nuclear_Power", type="Thermal", label="Nuclear Generation")

        # Relationships
        self.graph.add_edge("TTF_Gas", "DE_LU", relation="DRIVES_MARGINAL_COST")
        self.graph.add_edge("EU_ETS_Carbon", "DE_LU", relation="INCREASES_THERMAL_COST")
        self.graph.add_edge("Wind_Power", "DE_LU", relation="DAMPENS_SPOT_PRICE")
        self.graph.add_edge("Nuclear_Power", "FR", relation="BASELOAD_SUPPLY")
        self.graph.add_edge("FR", "DE_LU", relation="INTERCONNECTOR_EXPORT")
        self.graph.add_edge("DE_LU", "NL", relation="INTERCONNECTOR_EXPORT")

    def add_relationship(
        self,
        source: str,
        target: str,
        relation: str,
        source_type: str = "Event",
        target_type: str = "GridZone",
    ) -> None:
        """Adds a dynamic entity relationship to the Knowledge Graph."""
        if source not in self.graph:
            self.graph.add_node(source, type=source_type, label=source)
        if target not in self.graph:
            self.graph.add_node(target, type=target_type, label=target)
        self.graph.add_edge(source, target, relation=relation)

    def query_subgraph_context(self, entities: list[str], max_hops: int = 2) -> list[str]:
        """
        Extracts multi-hop graph subgraphs for target entities and formats human-readable context.
        """
        relations_found: list[str] = []
        visited_edges: set[tuple[str, str]] = set()

        for entity in entities:
            if entity not in self.graph:
                continue

            # Traverse outgoing and incoming edges up to max_hops
            nodes_to_visit = {entity}
            for _ in range(max_hops):
                next_nodes = set()
                for n in nodes_to_visit:
                    # Outgoing edges
                    for neighbor in self.graph.successors(n):
                        edge_key = (n, neighbor)
                        if edge_key not in visited_edges:
                            visited_edges.add(edge_key)
                            rel_type = self.graph.edges[n, neighbor].get("relation", "CONNECTED_TO")
                            relations_found.append(f"{n} --[{rel_type}]--> {neighbor}")
                            next_nodes.add(neighbor)
                    # Incoming edges
                    for predecessor in self.graph.predecessors(n):
                        edge_key = (predecessor, n)
                        if edge_key not in visited_edges:
                            visited_edges.add(edge_key)
                            rel_type = self.graph.edges[predecessor, n].get(
                                "relation", "CONNECTED_TO"
                            )
                            relations_found.append(f"{predecessor} --[{rel_type}]--> {n}")
                            next_nodes.add(predecessor)
                nodes_to_visit = next_nodes

        return relations_found

    def get_summary_context(self, market_symbol: str) -> str:
        """Returns structured GraphRAG summary text for LLM agent prompts."""
        rel_list = self.query_subgraph_context([market_symbol], max_hops=2)
        if not rel_list:
            return f"No GraphRAG context relationships found for {market_symbol}."
        return f"GraphRAG Knowledge Context for {market_symbol}:\n" + "\n".join(
            f"- {r}" for r in rel_list
        )


_GRAPH_INSTANCE: EnergyKnowledgeGraph | None = None


def get_knowledge_graph() -> EnergyKnowledgeGraph:
    """Returns singleton EnergyKnowledgeGraph instance."""
    global _GRAPH_INSTANCE
    if _GRAPH_INSTANCE is None:
        _GRAPH_INSTANCE = EnergyKnowledgeGraph()
    return _GRAPH_INSTANCE


def query_graph_context(market_symbol: str) -> str:
    """Convenience helper for querying Knowledge Graph RAG context."""
    kg = get_knowledge_graph()
    return kg.get_summary_context(market_symbol)
