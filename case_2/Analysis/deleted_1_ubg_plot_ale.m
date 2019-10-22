% deleted_1_ubg_plot from cal_Vertical_Transp_ale
redblu=cbrewer('div', 'RdBu',15);
% bird view
salt = ncread(fn,'salt',[1 1 1 100],[inf inf inf 1]); 
figure('position', [896 3 385 702]);
pcolor(xr./1000, yr./1000, salt(:,:,10))
colorbar, ca = max(abs(caxis));
% colormap(gca, flipud(redblu))
% caxis([-ca ca])
shading flat
% quick commands to plot;
xtrani = yr(1,:); xtran = xtrani(ones(30,1),:)';
% depth = squeeze(grd.z_r(loc_bg,:,:));
% xlim([-40 10 ]);
% ylim([-50 100])
line([xlim],[30 30], 'color', [0 0 0]);