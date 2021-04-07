function [mapped_pose] = mpii_map_to_gt_bone_lengths(pred, gt, o1, traversal_order)
mapped_pose = pred;

for i = 1:length(traversal_order)
    idx = traversal_order(i);
    gt_bone_length = norm(gt(:,idx) - gt(:,o1(idx)));
    pred_bone_vector = (pred(:,idx) - pred(:,o1(idx)));
    pred_bone_vector = pred_bone_vector * gt_bone_length / norm(pred_bone_vector);
    mapped_pose(:,idx) = mapped_pose(:,o1(idx)) + pred_bone_vector;
end


end
