function output = get_dim_and_tr(tree_node, output)

if(nargin < 2)
  output = [];
end

if(length(tree_node.parent) > 0)
  output = get_dim_and_tr(tree_node.parent, output);
end

output(end+1) = tree_node.dim;

if( length(tree_node.right_constrain) > 0)
  output(end+1) = tree_node.right_constrain;
  output(end+1) = -1;
elseif( length(tree_node.left_constrain) > 0)
  output(end+1) = tree_node.left_constrain;
  output(end+1) = +1;
end
