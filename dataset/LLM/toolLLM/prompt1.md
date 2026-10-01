
### **Task Description**

---

You are tasked with analyzing a user's natural language intent (`userIntent`) and converting it into structured outputs. Your task involves identifying the user's intents, determining the corresponding requirements, and converting the user's natural language intent about a network slice (virtual network) into a GML-formatted network descriptor.

The Graph Modelling Language (GML) format, describing a virtual network slice, should adhere to the rules and examples provided.
### **Reference**
#### GML Format

The GML format consists of the following elements:

- `description`: A brief text describing the network slice.
- `num_nodes`: The total number of nodes in the graph.
- `node`: Represents a node in the network, with attributes:
  - `id`: A unique identifier for the node.
  - `label`: A string label for the node, same as `id`.
  - `cpu`: The computational resources allocated to the node, measured in units.
- `edge`: Represents a connection between two nodes, with attributes:
  - `source`: The source node ID.
  - `target`: The target node ID.
  - `bw`: The bandwidth resources allocated to the edge, measured in units.

#### Example GML

The following GML defines a virtual network slice with 4 nodes and 4 links:
graph [
  description "this is a slice"
  num_nodes 4
  node [
    id 0
    label "0"
    cpu 10
  ]
  node [
    id 1
    label "1"
    cpu 19
  ]
  node [
    id 2
    label "2"
    cpu 1
  ]
  node [
    id 3
    label "3"
    cpu 50
  ]
  edge [
    source 0
    target 1
    bw 49
  ]
  edge [
    source 1
    target 2
    bw 44
  ]
  edge [
    source 1
    target 3
    bw 19
  ]
  edge [
    source 2
    target 3
    bw 50
  ]
]

#### Resource Constraints and Considerations

**Resource Limits:**
The total computational power (cpu) for nodes and bandwidth (bw) for edges is limited to a range of [50, 100] units.
Nodes with high computational power should be allocated more CPU units, while edges with high bandwidth should be allocated more bandwidth units.

**Defining "Resources Level":**
A node with cpu ≥ 25 units can be considered "high-computation."
A node with cpu ≥ 10 && cpu < 25 units can be considered "moderate-computation."
A node with cpu < 10 units can be considered "low-computation."
A link with bw ≥ 25 units can be considered "high-bandwidth."
A link with bw ≥ 10 && bw < 25 units can be considered "moderate-bandwidth."
A link with bw < 10 units can be considered "low-bandwidth."
The bandwidth resource level of the entire slice depends on the link with the lowest bandwidth, so usually the links have the same bandwidth.

**Inferring Topology and Resources:**
Analyze the user's intent for possible topological requirements (e.g., linear, star, mesh).
Determine the number of nodes and links based on the user's high-level description.
Distribute resources judiciously to balance computational and bandwidth requirements across the slice.
The entire network is a connected graph, so there should be no isolated nodes
If the topology of the slice is difficult to infer, or the user does not specify the topology, you can use the default topology, a ring topology with three nodes and three edges.

---

### **Historical Examples**
#### Example 1:

User Intent:
"I need a high-performance network slice with 3 main nodes connected in a star topology. The central node should handle most of the computation, while the links require moderate bandwidth."

Output:

```python
{
    gml:"graph [
    description "high-performance star network"
    num_nodes 3
    node [
        id 0
        label "0"
        cpu 45
    ]
    node [
        id 1
        label "1"
        cpu 25
    ]
    node [
        id 2
        label "2"
        cpu 25
    ]
    edge [
        source 0
        target 1
        bw 10
    ]
    edge [
        source 0
        target 2
        bw 10
    ]
    ]"
}
```
#### Explanation for Example 1:

- The user requested a star topology with a high-performance central node
- CPU resources were allocated to the central node (id=0, cpu=45 units) and evenly distributed among the leaf nodes (25 units each), aligning with the user's desire of "high-performance."
- Links were assigned moderate bandwidth (10 units each), aligning with the user’s description of "moderate bandwidth."

---

### **Task Requirements**
1. Ensure the output adheres to GML format.
2. Handle vague or incomplete intents by making reasonable assumptions based on historical examples and provided constraints.
3. Ensure that the resources allocated in generated gml file meets the given constraints and reflects the user’s intent.

---

### **Given User Intent**
I want a slice with low cpu and high bw, it should contain 5 nodes

### **Your Output**  
Please generate the list of tuples based on the intent.