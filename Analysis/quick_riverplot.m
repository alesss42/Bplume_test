% Show river change

figure
oceantm = ncread('ocean_his_0001.nc', 'ocean_time');
s = ncread('ocean_his_0001.nc', 'salt', [1,69,30,1], [inf, 1, 1, inf]);
plot(oceantm./86400, squeeze(s(end-1,1,1,:)))
xlim([0 40])
title('Time series of river salinity at the source')
xlabel('time in days')
figure
% cd ../IN_files
% rt = ncread("rivers_tracer_ale.nc", 'river_time');
% r_tr = ncread("rivers_tracer_ale.nc", 'river_transport');


rt = ncread("rivers_tracer.nc", 'river_time');
r_tr = ncread("rivers_tracer.nc", 'river_transport');


plot(rt./86400, sum(r_tr))
plot(rt./86400, r_tr(1,:))
title('Time series of river transport at source')
xlabel('time in days')
xlim([0 40])
