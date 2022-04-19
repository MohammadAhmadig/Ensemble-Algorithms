function tree_node = tree_node_w(max_split)

tree_node.left_constrain  = [];
tree_node.right_constrain = [];
tree_node.dim             = [];
tree_node.max_split       = max_split;
tree_node.parent         = [];

tree_node = class(tree_node, 'tree_node_w') ;