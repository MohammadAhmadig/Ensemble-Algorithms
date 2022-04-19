function y = calc_output(tree_node, XData)
y = XData(tree_node.dim, :) * 0 + 1;


for i = 1 : length(tree_node.parent)
  y = y .* calc_output(tree_node.parent, XData);
end

if( length(tree_node.right_constrain) > 0)
  y = y .* ((XData(tree_node.dim, :) < tree_node.right_constrain));
end
if( length(tree_node.left_constrain) > 0)
  y = y .* ((XData(tree_node.dim, :) > tree_node.left_constrain));
end