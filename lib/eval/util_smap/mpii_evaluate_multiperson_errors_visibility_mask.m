function [sequencewise_table] = mpii_evaluate_multiperson_errors_visibility_mask(sequencewise_error, joint_mask, joints)


% joints =  1:14;

sequencewise_pck = zeros(length(sequencewise_error),16);
%sequencewise_auc = zeros(length(sequencewise_error),15);
sequencewise_mpjpe = zeros(length(sequencewise_error),16);


 for i = 1:length(sequencewise_error)
     
     jm = joint_mask{i}(joints,1,:);
     masked_errors = sequencewise_error{i}(joints,1,:);
     masked_errors(isnan(masked_errors)) = 160;
     masked_errors = masked_errors .* jm;
     
     sequencewise_mpjpe(i,joints) = (sum(masked_errors,3) ./ (sum(jm,3)+0.0000000000000000000000000001))';
     sequencewise_mpjpe(i,length(joints)+1) = sum(masked_errors(:)) ./ sum(jm(:));
     
     sequencewise_pck(i,joints) = 1 - (sum(masked_errors>150,3) ./ (sum(jm,3)+0.0000000000000000000000001))';
     sequencewise_pck(i,length(joints)+1) = 1 - (sum(masked_errors(:)>150) ./ sum(jm(:)));
     sequencewise_pck(i,length(joints)+2) = sum(jm(:));
     
 end

 sequencewise_table = num2cell([sequencewise_mpjpe; sequencewise_pck]);