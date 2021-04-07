function [matching, matched] = mpii_multiperson_get_identity_matching(pose_2d, visibility, old_pose_2d, old_visibility, matching_threshold)

matching = zeros(1,length(visibility));
matched = zeros(1,length(old_visibility));

for i = 1:length(visibility)
    matching_score = zeros(1,length(old_visibility));
    for j = 1:length(old_visibility)
        if(matched(j)>0)
            continue;
        end
        diff = abs(pose_2d{i}-old_pose_2d{j});
        matches = (diff(1,:) < matching_threshold) & (diff(2,:) < matching_threshold);
        %Then count the ones where both are visible
        matching_score(j) = sum(matches((visibility{i}~=0) & (old_visibility{j}~=0)));
    end
    [score,value] = max(matching_score);
    if(score>0)
        matching(i) = value;
        matched(matching(i))=1;
    end
     
end

end