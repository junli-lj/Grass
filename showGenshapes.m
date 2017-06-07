function showGenshapes(genshapes)

for ii =1:length(genshapes)
    recover_boxes = genshapes{ii}.boxes;
    
    figure(ii);
    axis equal;
    axis([-0.7 0.7 -0.7 0.7 -0.7 0.7]);
    hold on;
    
    for jj = 1:size(recover_boxes,2)
        p = recover_boxes(:,jj);
        draw3dOBB_v2(p,'r');
    end
      
end