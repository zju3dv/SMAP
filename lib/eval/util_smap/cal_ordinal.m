function ordi = cal_ordinal(pd1, pd2, gt1, gt2, thres)
%     if pd1 - pd2 > thres
%         pd_ordi = 1;
%     elseif pd1 - pd2 < -thres
%         pd_ordi = -1;
%     else
%         pd_ordi = 0;
%     end
%     
%     if gt1 - gt2 > thres
%         gt_ordi = 1;
%     elseif gt1 - gt2 < -thres
%         gt_ordi = -1;
%     else
%         if ((gt1 - gt2) * (pd1 - pd2)) > 0
%             gt_ordi = sign(gt1 - gt2);
%             pd_ordi = gt_ordi;
%         else
%             gt_ordi = 0;
%         end
%     end
%     
%     if sign(pd_ordi) == sign(gt_ordi)
%         ordi = 1;
%     elseif sign(pd_ordi) == -sign(gt_ordi)
%         ordi = -1;
%     else
%         ordi = 0;
%     end
%%
%     if (gt1 - gt2) * (pd1 - pd2) > 0
%         ordi = 1;
%     else
%         if abs(gt1 - gt2) < thres
%             ordi = 0;
%         else
%             ordi = -1;
%         end
%     end
%% 
%     if (gt1 - gt2) * (pd1 - pd2) > 0 && abs(gt1 - gt2) >= thres && abs(pd1 - pd2) >= thres
%         ordi = 1;
%     else
%         if abs(gt1 - gt2) < thres && abs(pd1 - pd2) < thres
%             ordi = 0;
%         else
%             ordi = -1;
%         end
%     end

%%    
    if (gt1 - gt2) * (pd1 - pd2) > 0
        ordi = 1;
    else
        if abs(gt1 - gt2) < thres && abs(pd1 - pd2) < thres
            ordi = 0;
        else
            ordi = -1;
        end
    end
    
end