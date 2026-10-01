- ### **Historical Examples**

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