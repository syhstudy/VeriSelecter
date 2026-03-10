# VeriSelecter
Quality over Quantity: Diversity-Aware Data Selection for Efficient Verilog Code Generation



### Feature Dimension
We consider a total of 109-dimensional structural features, including 57 dimensions for the AST, 28 dimensions for the CFG, and 24 dimensions for the Netlist. The selection rules are presented in the table below.

| Representation Method | Feature Dimension | Feature Composition Details | Core Nodes & Description |
| :--- | :--- | :--- | :--- |
| **AST** | 57 | Node type dimension (54 categories) + depth parameters (2) + total node count (1) + node count per depth (depth+1) | Core Nodes: 'Source', 'ModuleDef', 'Decl', 'Input', 'Output', 'Reg', 'Wire','Assign', 'Always', 'Block', 'IfStatement', 'CaseStatement', 'Case', 'NonblockingSubstitution', 'BlockingSubstitution', 'Identifier', 'IntConst','Partselect', 'Pointer', 'Lconcat', 'Plus', 'Minus', 'Times', 'Divide', 'Mod', 'Power', 'Ulnot', 'Unot', 'Uand', 'Unand', 'Uor', 'Unor', 'Uxor', 'Uxnor', 'Sll', 'Srl', 'Sra', 'LessThan', 'GreaterThan', 'LessEq', 'GreaterEq', 'Eq', 'NotEq', 'Eql', 'NotEql', 'And', 'Xor', 'Xnor', 'Or', 'Land', 'Lor', 'Lnot' <br> Description: Covers all syntax nodes of hardware circuit, including modules, ports, registers/wires, assignments, conditional branches, arithmetic/logic/bitwise operations, etc. |
| **CFG** | 28 | Basic features (10 categories) + label features (len(common_labels) categories) + kernel features (n_iter+1 categories) | Core Nodes: 'Source', 'ModuleDef', 'Decl', 'Input', 'Output', 'Reg', 'Wire','Assign', 'Always', 'Block', 'IfStatement', 'CaseStatement', 'Identifier', 'IntConst', 'Plus', 'Minus', 'Times' <br> Description: Focuses on key control flow labels, including module entry, port definition, assignment logic, sequential blocks, conditional/branch statements and basic operation nodes. |
| **Netlist** | 24 | Basic features (9 categories) + type features (6 categories) + gate features (9 categories) | Core Nodes: 'model', 'input', 'output', 'gate', 'wire', 'latch', 'AND', 'OR', 'XOR', 'NAND', 'NOR', 'XNOR', 'BUF', 'NOT', 'LUT' <br> Description: Oriented to hardware netlist entities and gate-level logic, including model/input/output/wire/latch types, and AND/OR/NOT/XOR gate-level units, etc. |
