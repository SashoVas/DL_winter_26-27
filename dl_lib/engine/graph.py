import graphviz

from dl_lib.engine.value import Value


def trace(root: Value):

    nodes = set()
    children = set()
    for prev in root._prev:
        next_nodes, next_children = trace(prev)
        nodes |= next_nodes
        children |= next_children
        children.add((prev, root))

    nodes.add(root)

    return nodes, children


def draw_dot(root: Value):
    nodes, children = trace(root)
    dot = graphviz.Digraph(graph_attr={"rankdir": "LR"})

    for node in nodes:
        dot.node(
            name=node.label,
            label="{ "
            + f"{node.label} | data: {node.data:.4f} | grad: {node.grad:.4f}"
            + " }",
            shape="record",
        )

        if node._op:
            dot.node(name=node.label + node._op, label=node._op)
            dot.edge(node.label + node._op, node.label)

    for child, parent in children:
        dot.edge(child.label, parent.label + parent._op)

    return dot


def top_sort(root: Value) -> list:
    stack = []
    visited = set()

    def dfs(node: Value) -> None:
        if node in visited:
            return
        visited.add(node)
        for child in node._prev:
            dfs(child)
        stack.append(node)

    dfs(root)
    return stack
