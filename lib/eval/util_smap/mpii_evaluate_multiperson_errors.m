function [sequencewise_table] = mpii_evaluate_multiperson_errors(sequencewise_error, eval_joints)

joint_groups = mpii_get_pck_auc_joint_groups();
[~,~,~,joint_names] = mpii_get_joints('relevant');
joint_names = joint_names(1:15); 
num_joints = length(eval_joints); 
joint_names = joint_names(eval_joints);
all_errors = [];
sequencewise_pck = {};
sequencewise_auc = {};
sequencewise_mpjpe = cell(length(sequencewise_error)+1,num_joints+2);
sequencewise_mpjpe(1,2:(num_joints+1)) = joint_names;
sequencewise_mpjpe{1,(num_joints+2)} = 'Average';
 %Generate MPJPE and PCK/AUC By sequence first
 %error_dat = {};
 %delete('error_dat');
 for i = 1:length(sequencewise_error)
     if(isempty(all_errors))
         all_errors = sequencewise_error{i}(1:num_joints,1,:);
     else
         all_errors = cat(3,all_errors, sequencewise_error{i}(1:num_joints,1,:));
     end
     %all_activities = [all_activities; sequencewise_activity{i}(:)];
         
     error_dat(i) = mpii_3D_error(['TestSeq' int2str(i)], sequencewise_error{i}(1:num_joints,1,:));
     sequencewise_mpjpe{i+1,1}= ['TestSeq' int2str(i)];
     mpjpe = mean(sequencewise_error{i}(1:num_joints,1,:),3);
     sequencewise_mpjpe(i+1,2:(1+num_joints)) = num2cell(mpjpe');
     sequencewise_mpjpe{i+1,(2+num_joints)} = mean(mpjpe(:));
 end
 [pck, auc] = mpii_compute_3d_pck(error_dat, joint_groups, []);
 sequencewise_pck = [sequencewise_pck; pck];
 sequencewise_pck{1,1} = 'PCK';
 sequencewise_auc = [sequencewise_auc; auc];
 sequencewise_auc{1,1} = 'AUC';
     
sequencewise_table = sequencewise_mpjpe;
sequencewise_table(size(sequencewise_table,1)+1:size(sequencewise_table,1)+size(sequencewise_pck,1),1:size(sequencewise_pck,2)) = sequencewise_pck;
sequencewise_table(size(sequencewise_table,1)+1:size(sequencewise_table,1)+size(sequencewise_auc,1),1:size(sequencewise_auc,2)) = sequencewise_auc;
     
end