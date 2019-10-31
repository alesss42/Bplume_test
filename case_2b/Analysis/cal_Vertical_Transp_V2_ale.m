
% Estimating the transport at a given transect

clearvars, 
% close all
redblue = cbrewer('div','RdBu', 21);
load Mycolorbars.mat

%%
file_01  = '../OUT/ocean_his_0001.nc';
file_river = '../IN_files/rivers_tracer.nc';
fn       = [file_01];

% Initial conditions
salt_0 = ncread(fn, 'salt', [1,1,15,1], [inf, inf, 1, 1]);
s_ocn    = nanmean(nanmean(salt_0));
s_bc     = 33.9;

grd      = get_roms_grid(fn, fn, 1);
[ysize,xsize] = size(grd.x_rho);

time = ncread(fn,'ocean_time');
Dt   = ncread(fn,'dt');       % time step in seconds
nHIS = ncread(fn,'nHIS');   % time-step interval to record a his file

Vr = nc_vinfo(fn,'s_rho');
zsize = Vr.Size;
dx = 1./grd.pm;
dy = 1./grd.pn;
% load('ImpingingAngle_Qr1E3W10H5S1E3dRho10.mat'); % Previousely calculated

%%
% unix(['mkdir ./anim_temp_w']);
% ff = figure('paperunits','inches','paperposition',[0 0 12.5 7.5]);
%%  This seems to be the selection of time?
t_p_s = 1;
[~,t_idx_s] = min(abs(time/86400 - t_p_s));
t_p_e = 40.0;
[~,t_idx_e] = min(abs(time/86400 - t_p_e));
t_range = t_idx_s:t_idx_e;
tsize = length(t_range);
%%
i_mouth    = 111; %acroos shelf  % CHECK!!!!! because the file says 140
j_channel1 = 69;  % along shelf

hmax_plot  = 25; % Maximum depth

xr      = grd.x_rho - grd.x_rho(i_mouth, j_channel1);
yr      = grd.y_rho - grd.y_rho(i_mouth, j_channel1);

%%
%%%%% load surface salinity
s1_xy = ncread(fn,'salt',[1 1 zsize t_idx_s],[inf inf 1 tsize]);       % top  (t,y,x)

%%%%% interpolate to fine grid
[xi_i,yi_i] = meshgrid(-40e3:1e2:0,-15e3:1e3:60e3);
xi = xi_i'; yi =yi_i';
% x_transect  = 0;
%% I think this will never change because this gives you the place were the buldge is
%  [aa,i_mouth] = min( abs( xr(:,end) - x_transect) );
% cc_r  = [24:0.5:34];
% cc_u = [-0.3:0.02:0.3];
% cc_v = [-0.3:0.02:0.3];
% cc_w = [-40:2:40]*1e-5;
% %dim_1 = [-40 10 -10 200];
% dim_1 = [-30 2 -hmax_plot 1];
% dim_2 = [-30 2 -20 70];
% xxr = repmat(xr(:,1),[zsize   1]);
% xxw = repmat(xr(:,1),[zsize+1 1]);
% dx = permute(dx,[1 2 3]); dx = repmat(dx,[ 1 1 zsize]);
% dy = permute(dy,[1 2 3]); dy = repmat(dy,[ 1 1 zsize]);

%% Budget with out interpolation


%% Section copy from another code

dxi(:,:) = interp2(yr,xr,dx,yi,xi);
dyi(:,:) = interp2(yr,xr,dy,yi,xi);

%Select the location of the transect

t_loc2 =  40000;
t_loc3 =  0;
t_loc2_B =-25000;
%%

f1 = figure('Position', [618     3   985   880]);
f2 = figure('Position', [606     3   997   375]);
f3 = figure('Position', [1013           3         590         942]);
%%
tt_m = 1;



for tt =  t_idx_s:1:t_idx_e
salt = ncread(fn,'salt',[1 1 1 tt],[inf inf inf 1]); 
v = ncread(fn,'v',[1 1 1 tt],[inf inf inf 1]); 
zeta = ncread(fn,'zeta',[1 1  tt],[inf inf 1]); 
zr = set_depth(grd.Vtransform,grd.Vstretching,grd.theta_s,grd.theta_b,grd.hc,zsize,1,grd.h,zeta,0);
zw = set_depth(grd.Vtransform,grd.Vstretching,grd.theta_s,grd.theta_b,grd.hc,zsize,5,grd.h,zeta,0);
dz = permute(zw(:,:,2:end)-zw(:,:,1:end-1),[1 2 3]);

% adjusting v to salt points 
v_s = salt*0;
v_s(:,2:end-1,:) = 0.5*(v(:,1:end-1,:) + v(:,2:end,:));

for zz = 1:zsize
% This seems to be to generate a super high resolution of the bulge
si(:,:,zz) = interp2(yr,xr,squeeze(salt(:,:,zz)),yi,xi);
vi(:,:,zz) = interp2(yr,xr,squeeze(v_s(:,:,zz)),yi,xi);
dzi(:,:,zz) = interp2(yr,xr,squeeze(dz(:,:,zz)),yi,xi);     
zi(:,:,zz) = interp2(yr,xr,squeeze(zr(:,:,zz)),yi,xi);   
end

tn = find(yi(2,:)<=t_loc2, 1, 'last');
tnt = find(yr(2,:)<=t_loc2, 1, 'last');
tntb = find(yr(2,:)<=t_loc2_B, 1, 'last');

% %looking at a transect in higher resolution
si_tr =squeeze(si(:,tn,:));
vi_tr =squeeze(vi(:,tn,:));
zi_tr = squeeze(zi(:,tn,:));
xi_tr = xi(:, tn); xm_tr = xi_tr(:, ones(1,30));
dxi_tr = dxi(:,tn); dxm_tr = dxi_tr(:, ones(1,30));
dzi_tr = dzi(:,tn);

% %looking at a transect in low resolution
sr_tr_a =squeeze(salt(:,tnt,:));
vr_tr_a =squeeze(v(:,tnt,:));
zr_tr_a = squeeze(zr(:,tnt,:));
x_tr_a = xr(:, tnt); xmr_tr = x_tr_a(:, ones(1,30));
dx_tr = dx(:,tnt); dxmr_tr = dx_tr(:, ones(1,30));
dz_tr = dz(:,tnt);

sr_tr_b =squeeze(salt(:,tntb,:));
vr_tr_b =squeeze(v(:,tntb,:));
zr_tr_b = squeeze(zr(:,tntb,:));
x_tr_b = xr(:, tnt); xmr_trb = x_tr_b(:, ones(1,30));
dx_trb = dx(:,tnt); dxmr_trb = dx_trb(:, ones(1,30));
dz_trb = dz(:,tnt);


%% Salinity boundary
ind_plot = mod(tt,20);
if ind_plot ==0
figure(f2)
% scatter(xm_tr(:), zi_tr(:),20, vi_tr(:), 'fill'); hold on
contourf(xm_tr./1000, zi_tr, vi_tr, 21, 'linestyle', 'none'); hold on
caxis([-.08 .08]) % m/s
c = colorbar; colormap(flipud(redblue))
c.Label.String = 'Along-shelf velocity v (m/s)';
contour(xm_tr./1000, zi_tr, si_tr, [32:.2:34], 'k',  'linewidth', 2)
[cg hg]=contour(xm_tr./1000, zi_tr, si_tr, [33.8 33.8], 'g',  'linewidth', 4);
[c h]= contour(xm_tr./1000, zi_tr, vi_tr, [0 0],'r',  'linewidth', 5);
legend([hg h], 's = 33.8', 'v = 0', 'location','SouthWest')
xlim([-25 0])
set(gca, 'color', [.5 .5 .5])
M2(tt_m) = getframe(f2);
hanles_font = findall(0,'type','text');
set(hanles_font, 'interpret', 'latex');

figure(f3)
redblu=cbrewer('div', 'RdBu',15);
% bird view
pcolor(xr./1000, yr./1000, squeeze(salt(:,:,30)))
colorbar, ca = max(abs(caxis));
% colormap(gca, flipud(redblu))
% caxis([-ca ca])
shading flat
% quick commands to plot;
xtrani = yr(1,:); xtran = xtrani(ones(30,1),:)';
% depth = squeeze(grd.z_r(loc_bg,:,:));
xlim([-40 10 ]);
ylim([-50 100])
line([xlim],[t_loc2 t_loc2]./1000, 'color', [0 0 0]);
line([xlim],[t_loc2_B t_loc2_B]./1000, 'color', [0 0 0]);
M3(tt_m) = getframe(f3);
title(['Day ', num2str(time(tt)./(60*60*24))])
end



%% Estimated transport
s_bc = 33.8;  % Target "salinity" surface

% Creating the mask
 mask = si_tr * 0;
 mask(si_tr <= s_bc) = 1;
 
% Transport only "fresh water" out
dxi_m_pl = dxi_tr.*mask;
dzi_m_pl = dzi_tr.*mask;
vi_tr_pl = vi_tr.*mask;
% Stamting area
Ai_pl = dxi_m_pl.*dzi_m_pl;
A_m2_pl = sum(sum(Ai_pl));
A_km2_pl = A_m2_pl./(1000*1000);

Volume_flux_pl(:,:,tt) = (vi_tr_pl.*(Ai_pl));
Tf_pl(tt) = squeeze(nansum(nansum(Volume_flux_pl(:,:,tt))));


% Total transport trhough boundary A
Ai_a = dx_tr.*dz_tr;
Ai_b = dx_trb.*dz_trb;
A_m2_a = sum(sum(Ai_a));
A_m2_b = sum(sum(Ai_b));
A_km2_a = A_m2_a./(1000*1000);
A_km2_b = A_m2_b./(1000*1000);

Qa(:,:,tt) = (vr_tr_a.*(Ai_a));
Qb(:,:,tt) = (vr_tr_b.*(Ai_b));
Qf_a(:,:,tt) = Qa(:,:,tt)./(s_ocn./(s_ocn-sr_tr_a));
Qf_b(:,:,tt) = Qb(:,:,tt)./(s_ocn./(s_ocn-sr_tr_b));

Tf_a(tt) = squeeze(nansum(nansum(Qf_a(:,:,tt))));
Tf_b(tt) = squeeze(nansum(nansum(Qf_b(:,:,tt))));
Tf_ab(tt) = Tf_a(tt)+Tf_b(tt);
% Tf_t(tt) = squeeze(nansum(nansum(Volume_flux_t(:,:,tt))));
 
% Tf_river = ncread(file_river, 'river_transport');
%%
load Quick_river_info
Qr = -nansum(river_transport,2)./(s_ocn./(s_ocn-24))
%% Salinity boundary

if ind_plot == 0 
vlim = 5;
figure(f1)
subplot(311)
contourf(xmr_tr./1000, zr_tr_a, squeeze(Qf_a(:,:,tt)), 21, 'linestyle', 'none'); hold on
[cg hg]=contour(xm_tr./1000, zi_tr, si_tr, [33.8 33.8], 'g',  'linewidth', 4);
c = colorbar; colormap(flipud(redblue))
c.Label.String = 'Volument flux (m^3/s)';
% title('Total volume flux throught cross-shelf transect')
title('Volume flux of freshwater north transect')
caxis([-vlim vlim])
xlim([-50 5])
set(gca, 'color', [.5 .5 .5])

subplot(312)
contourf(xmr_trb./1000, zr_tr_b, squeeze(Qf_b(:,:,tt)), 21, 'linestyle', 'none'); hold on
% [cg hg]=contour(xm_tr./1000, zi_tr, si_tr, [33.8 33.8], 'g',  'linewidth', 4);
c = colorbar; colormap(flipud(redblue))
set(gca, 'color', [.5 .5 .5])
caxis([-vlim vlim])
xlim([-50 5])
c.Label.String = 'Volument flux (m^3/s)';
% title('Volume flux of Costal Current throught cross-shelf transect')
title('Volume flux of freshwater south transect')

subplot(313)
L1= plot(time(1:tt)./(60*60*24), Tf_a, 'k', 'LineWidth', 2); hold on
L2 =plot(time(1:tt)./(60*60*24), Tf_b, 'r', 'LineWidth', 2); hold on
L3 =plot(time(1:tt)./(60*60*24), Tf_ab, 'g', 'LineWidth', 2); hold on
R1 = plot(river_time./(60*60*24),Qr, 'b','LineWidth', 4); 
title('Freshwater budget');
legend([L1, L2, L3, R1] , 'North', 'South', 'Total','River' , 'Location', 'NorthWest');
line(xlim, [0 0], 'linestyle', '--', 'color', 'b')
xlabel('Time (days)')
ylabel('Transport $m^3/s$')
ylim([-.5e3 4e3])
hanles_font = findall(0,'type','text');
set(hanles_font, 'interpret', 'latex');
M(tt_m) = getframe(f1);
tt_m = tt_m+1;
    
    % vlim = 30;
% figure(f1)
% subplot(311)
% contourf(xm_tr./1000, zi_tr, squeeze(Volume_flux_t(:,:,tt)), 21, 'linestyle', 'none'); hold on
% [cg hg]=contour(xm_tr./1000, zi_tr, si_tr, [33.8 33.8], 'g',  'linewidth', 4);
% c = colorbar; colormap(flipud(redblue))
% c.Label.String = 'Volument flux (m^3/s)';
% title('Total volume flux throught cross-shelf transect')
% caxis([-vlim vlim])
% set(gca, 'color', [.5 .5 .5])
% subplot(312)
% contourf(xm_tr./1000, zi_tr, squeeze(Volume_flux_pl(:,:,tt)), 21, 'linestyle', 'none'); hold on
% [cg hg]=contour(xm_tr./1000, zi_tr, si_tr, [33.8 33.8], 'g',  'linewidth', 4);
% c = colorbar; colormap(flipud(redblue))
% set(gca, 'color', [.5 .5 .5])
% caxis([-vlim vlim])
% c.Label.String = 'Volument flux (m^3/s)';
% title('Volume flux of Costal Current throught cross-shelf transect')
% 
% subplot(313)
% L1= plot(time(1:tt)./(60*60*24), Tf_t, 'k', 'LineWidth', 2); hold on
% plot(time(tt)./(60*60*24), Tf_t(tt), 'rx'); hold on
% 
% L2 =plot(time(1:tt)./(60*60*24), Tf_pl, 'r', 'LineWidth', 2); hold on
% legend([L1, L2] , 'Total Flux', 'CC flux', 'Location', 'SouthEast');
% xlabel('Time (days)')
% ylabel('Transport $m^3/s$')
% ylim([0 8e4])
% hanles_font = findall(0,'type','text');
% set(hanles_font, 'interpret', 'latex');
% M(tt_m) = getframe(f1);
% tt_m = tt_m+1;
end



% caxis([-100 100])
% figure
% contourf(xm_tr./1000, zi_tr, vi_tr, 21, 'linestyle', 'none'); hold on
% colorbar; colormap(flipud(redblue))

% pt = 20;
% x_bc=nan(t_idx_e,1600);
% y_bc=nan(t_idx_e,1600);
% ang_correct=nan(t_idx_e,1600);
% 
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% start plotting %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% for tt = t_idx_e
% s_top = ncread(fn,'salt',[1 1 zsize tt],[inf inf 1 1]); 
% s_t = ncread(fn,'salt',[1 1 1 tt],[inf inf inf 1]); 
% u_top = ncread(fn,'u', [1 1 zsize tt],[inf inf 1 1]); 
% v_top= ncread(fn,'v',[1 1 zsize tt],[inf inf 1 1]); 
% w_t =  ncread(fn,'w',[1 1 1  tt],[inf inf inf 1]); 
% zeta = ncread(fn,'zeta',[1 1  tt],[inf inf 1]); 
% 
% zr = set_depth(grd.Vtransform,grd.Vstretching,grd.theta_s,grd.theta_b,grd.hc,zsize,1,grd.h,zeta,0);
% zw = set_depth(grd.Vtransform,grd.Vstretching,grd.theta_s,grd.theta_b,grd.hc,zsize,5,grd.h,zeta,0);
% dz = permute(zw(:,:,2:end)-zw(:,:,1:end-1),[1 2 3]);
% 
% % interpret u.v into rho-coordinate
% u_top_rho = s_top*0; v_top_rho = s_top*0;                    
% u_top_rho(2:end-1,:) = 0.5*(u_top(1:end-1,:) + u_top(2:end,:));
% v_top_rho(:,2:end-1) = 0.5*(v_top(:,1:end-1) + v_top(:,2:end));
% ui = interp2(yr,xr,u_top_rho,yi,xi);
% vi = interp2(yr,xr,v_top_rho,yi,xi);
% %%
% ubg = ncread(fn,'u',   [i_mouth-30 1 1 tt ],[2 inf inf 1]);  
% ubg = squeeze( nanmean(ubg,1) );
% loc_bg = i_mouth; 
% % Deleted plot to undesrand acroos shelf flow
%  deleted_1_ubg_plot
% %%
% y_transect  = y_impinge(tt) + 5e3;
% [aa,jj_cc1] = min( abs( yr(end,:) - y_transect) );
% 
% vcc = ncread(fn,'v',   [1 jj_cc1-1 1 tt],[inf 2 inf 1]);  
% vcc = squeeze( nanmean(vcc,2) );
% %uu = s1 * 0; uu(:,2:end-1) = 0.5*(u1(:,1:end-1)+u1(:,2:end)); clear u1;
% %u1 = uu; clear uu;
% 
% w1 = ncread(fn,'w', [ 1 jj_cc1 1 tt],[inf 1 inf 1]);  
% zr1= squeeze(zr(:,jj_cc1,:))';
% zw1= squeeze(zw(:,jj_cc1,:))';
% 
% % cal transport into bulge and from bulge to coastal current
% mask = s_t; 
% mask(s_t > s_bc) = 0; mask(s_t <= s_bc) = 1; 
% mask(i_mouth:end,:,:)  = 0;
% mask_a = s_t;
% mask_a(s_t <= s_bc) = 0; mask_a(s_t > s_bc) = 1; 
% mask_a(i_mouth:end,:,:) = nan;
% 
% transp_bg(tt) =  nansum(nansum(squeeze(dy(:,i_mouth,:)).*ubg.*squeeze(dz(:,i_mouth,:))));
% transp_cc(tt) =  nansum(nansum(squeeze(dx(jj_cc1,:,:)) .*vcc.* squeeze(dz(jj_cc1,:,:)).* squeeze(mask(:,jj_cc1,:))  ));
% vdxdz1 = vcc.*squeeze(dx(jj_cc1,:,:)) .* squeeze(dz(jj_cc1,:,:)); 
% return_transp1(tt) = nansum(nansum(vdxdz1(vdxdz1<0).*mask_a(vdxdz1<0)));
% 
% 
% %cal vertical transport within bulge 
% w_rho_t = nan(141,308,30);
% 
% % dont understand this oone below
% Area_bulge = nansum(nansum(  dx(1:jj_cc1,:,zsize) .* dy(1:jj_cc1,:, zsize) ));
% for i=1:i_mouth-1
%     for j=4:308
%         w_rho_t(i,j,:) = interp1(squeeze(zw(j,i,:)),squeeze(w_t(i,j,:)),squeeze(zr(j,i,:)));
%     end;
% end;
% w_temp = w_rho_t .* mask; int_w = squeeze(sum(w_temp));
% L1 =  y_impinge(tt)-min(y_bc(tt,:));
% L2 =  max(x_bc(tt,:));
% 
% % Defining the sal
% 
% end
% %%
% figure;
% subplot(2,1,1)
% plot(time(t_idx_s:t_idx_e)/86400,-transp_cc(t_idx_s:t_idx_e)-transp_bg(t_idx_s:t_idx_e));
% title('W [m^3/s]')
% 
% subplot(2,1,2)
% plot(time(t_idx_s:t_idx_e)/86400,-return_transp1(t_idx_s:t_idx_e));
% title('return transport [m^3/s]')
