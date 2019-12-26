% M file to create the grid. This will be a combination of JW script with
% Julies. 
clearvars
fname = 'grid_case2_finergrid02_ale.nc';
%% ADDITION FROM JW
% for all use:
    spherical='F';
    projection='mercator';

%%

% construct ESTUARY grid.
L_east   =  0e3;
L_west   = 100e3;
pt_tot   = 200;
dx1      = 0.5e3; % Smallest grid size in the x directionv
dx2      =  10e3;
Lx_h_east = 200e3;
Lx_h_west = 50e3;

i_pt_east = round(Lx_h_east/dx1);
i_pt_west = round(Lx_h_west/dx1);
r_pt_east = 10;
r_pt_west = 10;

% Creates a dx than increses as tan to (dx2+dx1). There is no zero
i_temp    = [1:pt_tot];
ff_east   = 0.5 * ( tanh( (i_temp-(i_pt_east+r_pt_east))/r_pt_east ) + 1 );
dx_east   = ff_east*dx2 + dx1 ;
xx_east   = cumsum(dx_east); % Distance vectour in the east
[aa,i_eastbc] = min(abs(xx_east - L_east));

% Creates a dx than increses as tan to (dx2+dx1). There is no zero
ff_west   = 0.5 * ( tanh( (i_temp-(i_pt_west+r_pt_west))/r_pt_west ) + 1 );
dx_west   = ff_west*dx2 + dx1;
xx_west   = cumsum(dx_west);
[aa,i_westbc] = min(abs(xx_west - L_west));

% Combined
dx_r = [dx_west(i_westbc:-1:1) dx_east(1:100)];



%%
if 1
f1 = figure;
subplot(1,2,1)
plot(dx_east(1:i_eastbc),'b-o'); hold on
plot(dx_west(1:i_westbc),'r-o');
subplot(1,2,2)
plot(xx_east(1:i_eastbc),'b-');
hold on;
plot(xx_west(1:i_westbc),'r');

figure;
subplot(1,2,1)
plot(dx_r,'b-o');
subplot(1,2,2)
plot(cumsum(dx_r),'b-o');
end
%%
%%%%% north-south grid
L_north  = 500e3;
L_south  = 200e3;
pt_tot_north   = 600;
pt_tot_south   = 200;
dy1      = 1e3;  % Smallest grid size in the y direction
dy2      = 10e3;
Ly_h_north = 500e3;
Ly_h_south = 40e3;
j_pt_north = round(Ly_h_north/dy1);
j_pt_south = round(Ly_h_south/dy1);
r_pt_north = 20;
r_pt_south = 10;

% Creates a dx than increses as tan to (dx2+dx1). There is no zero
j_temp_n    = [1:pt_tot_north];
ff_north   = 0.5 * ( tanh( (j_temp_n-(j_pt_north+r_pt_north))/r_pt_north ) + 1 );
dy_north   = ff_north*dy2 + dy1;
yy_north   = cumsum(dy_north);
[aa,j_northbc] = min(abs(yy_north - L_north));



% Creates a dx than increses as tan to (dx2+dx1). There is no zero
j_temp_s    = [1:pt_tot_south];
ff_south   = 0.5 * ( tanh( (j_temp_s-(j_pt_south+r_pt_south))/r_pt_south ) + 1 );
dy_south   = ff_south*dy2 + dy1;
yy_south   = cumsum(dy_south);
[aa,j_southbc] = min(abs(yy_south - L_south));
dy_r = [dy_south(j_southbc:-1:1) dy_north(1) dy_north(1:325)];   % for symmetric
%%
if 1
f2 = figure;
subplot(1,2,1)
plot(dy_north(1:j_northbc),'b-o'); hold on
plot(dy_south(1:j_southbc),'r-o');
subplot(1,2,2)
plot(yy_north(1:j_northbc),'b-');
hold on;
plot(yy_south(1:j_southbc),'r');

figure;
subplot(1,2,1)
plot(dy_r,'b-o');
subplot(1,2,2)
plot(cumsum(dy_r),'b-o');
end
%% 
% This part is interesting, we get the dimension for the grid and we obtained 
% of the coastthe location of
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
filename = 'H5FlatW10S1E3_grid.nc';     % January 6th 2011
disp('The values for Lm and Mn  are:')
Lm = length(dx_r)-2
Mm = length(dy_r)-2
disp('Location of the coastline')
i_mouth = i_westbc
%j_channel = j_southbc+1
%channel_center = j_channel;
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Grid parameters.

L = Lm+1;
M = Mm+1;
Lp= L +1;
Mp= M +1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Construct grid, and calculate grid metrics.

% Construct dx_r and dy_r.  Integrate to get x_r and y_r.

disp(['Minimum dx = ',num2str(min(dx_r))]);
disp(['Minimum dy = ',num2str(min(dy_r))]);
% [pm,pn]=meshgrid(1./dx_r,1./dy_r);%
[pn,pm]=meshgrid(1./dy_r,1./dx_r);   % % ASR DIM CHANGE 

% Calculate x_r and y_r
dx=[dx_r(1)./2 0.5.*(dx_r(1:end-1)+dx_r(2:end))];
dy=[dy_r(1)./2 0.5.*(dy_r(1:end-1)+dy_r(2:end))];
% [x_r,y_r]=meshgrid(cumsum(dx),cumsum(dy)); % ASR DIM CHANGE 
[y_r,x_r]=meshgrid(cumsum(dy), cumsum(dx));

% Shift grid so x_u(:,1)=0 and y_v(1,:)=0.
% y_r = y_r - y_r(1,1) - (y_r(2,1)-y_r(1,1))/2;
% x_r = x_r - x_r(1,1) - (x_r(1,2)-x_r(1,1))/2;
y_r = y_r - y_r(1,1) - (y_r(1,2)-y_r(1,1))/2;
x_r = x_r - x_r(1,1) - (x_r(2,1)-x_r(1,1))/2; % ASR DIM CHANGE 



% Calculate dmde and dndx.
% dndx = zeros(Mp,Lp);
% dmde = zeros(Mp,Lp);
dndx = zeros(Lp,Mp);
dmde = zeros(Lp,Mp);

% dndx(2:M,2:L) = (1./pn(2:M,3:Lp) - 1./(pn(2:M,1:Lm)))/2;
% dmde(2:M,2:L) = (1./pm(3:Mp,2:L) - 1./(pm(1:Mm,2:L)))/2;
dndx(2:L,2:M) = (1./pn(3:Lp,2:M) - 1./(pn(1:Lm,2:M)))/2;
dmde(2:L,2:M) = (1./pm(2:L,3:Mp) - 1./(pm(2:L,1:Mm)))/2;

% % Calculate x_u, etc.
% x_u = (x_r(:,1:L) + x_r(:,2:Lp))/2;
% y_u = (y_r(:,1:L) + y_r(:,2:Lp))/2;
% x_v = (x_r(1:M,:) + x_r(2:Mp,:))/2;
% y_v = (y_r(1:M,:) + y_r(2:Mp,:))/2;
% x_p = (x_r(1:M,1:L) + x_r(2:Mp,2:Lp))/2;
% y_p = (y_r(1:M,1:L) + y_r(2:Mp,2:Lp))/2;
% % el = y_u(end,1);
% xl = x_v(1,end);

% Calculate x_u, etc. % Changed to make it ASR DIM change
x_u = (x_r(1:L,:) + x_r(2:Lp,:))/2;
y_u = (y_r(1:L,:) + y_r(2:Lp,:))/2;

x_v = (x_r(:,1:M) + x_r(:,2:Mp))/2;
y_v = (y_r(:,1:M) + y_r(:,2:Mp))/2;

x_p = (x_r(1:L,1:M) + x_r(2:Lp,2:Mp))/2;
y_p = (y_r(1:L,1:M) + y_r(2:Lp,2:Mp))/2;

el = y_u(1,end);
xl = x_v(end,1);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% find river points.

% 1st river pt is at south bc 
dL_river1 = 20e3;                        % 2nd river pt from 1st high res point 
jr = find(dy_r <= 1.01*dy1);      % high resolution region
% [val,j_river1] = min(abs(y_r(:,1) - (y_r(jr(1),1)+dL_river1)));   % 2nd river point 
% ASR
[val,j_river1] = min(abs(y_r(1,:) - (y_r(1,jr(1))+dL_river1)));   % 2nd river point 
%%
L0 = y_r(1,1) - y_r(1,jr(1));
L1 = y_r(1,jr(1)) - y_r(1,j_river1);
L2 = y_r(1,j_river1) - y_r(1,jr(end));
L3 = y_r(1,jr(end)) - y_r(1,end);
disp(['distance between south bc to 1st high-res point = ',num2str(-L0/1e3),' km'])
disp(['distance between 1st high-res point to 2nd river point source = ',num2str(-L1/1e3),' km'])
disp(['distance between 2nd river point source to last high-res point = ',num2str(-L2/1e3),' km'])
disp(['distance between last high-res point to north bc = ',num2str(-L3/1e3),' km'])

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Construct grid topography.
h_min         = 05;                  % coastal wall depth
h_channel_max = 05;                  % channel depth
slope_land    = 0;             % landward slope
slope_sea     = 1.0e-3;             % seaward slope


W_river1  = 10e3;        % width of river
[val,j_1] = min(abs(y_r(1,:) - (y_r(1,j_river1)-W_river1/2)));
[val,j_2] = min(abs(y_r(1,:) - (y_r(1,j_river1)+W_river1/2)));
j_river1_range = j_1:j_2;

%%
figure;
subplot(121)
plot(x_r(:,1),dx_r,'b-o'); title('east-west');
subplot(122)
plot(y_r(1,:),dy_r,'b-o'); title('north-south');
hold on
plot(y_r(1,jr(1)),dy_r(jr(1)),'m.','markersize',18);
plot(y_r(1,j_river1),dy_r(j_river1),'g.','markersize',18);
plot(y_r(1,j_river1_range),dy_r(j_river1_range),'g-','linewidth',4);
%%
% store river file

if 1
    socn = 34;
    ns1 = length(j_river1_range);
%     rout(1).X = (Lp - 1) * ones(ns1,1);
    rout(1).X = (Lp - 1) * ones(ns1,1);
%     rout(1).X = (Lp) * ones(ns1,1);
    rout(1).Y = j_river1_range-1;
    rout(1).D = 0 * ones(ns1,1); 
    rout(1).sig = -1 * ones(ns1,1);
    rout(1).id   =  1 * ones(ns1,1);
    rout(1).flag =  4 * ones(ns1,1);
    rout(1).salt =  24;
    rout(1).dye  =  1;
    rout(1).Qf   = 1000;
    rout(1).Q    = rout(1).Qf * socn/(socn - rout(1).salt);

    save('riverfile_ale','rout');
end

%% Make depth

Width_temp = zeros(1,Lp); Width_temp(:) = W_river1;

for i=1:Lp
    h_channel(i) = h_channel_max + slope_land * ( x_r(i_mouth,1) - x_r(i,1) );
end

h_channel(find(h_channel > h_channel_max)) = h_channel_max;

dh = h_channel - h_min;  % Gaussian channel
%dh = h_channel * 0;       % rectangular channel
%%
for i=1:Lp
for j=1:Mp

%h(j,i) = h_channel(i);       % rectangular channel
h(i,j) = h_min + dh(i) * exp(-( (y_r(i,j) - y_r(i,j_river1))/(0.250*Width_temp(i)) )^2 );  % Gaussian channel
end
end
%%
% -------- this creates offshore slope --------------------------------------
if 1
h_mouth_shoal = h(i_mouth,j_1) ;     % shoal depth at the mouth
depth_max = 150;

for j=1:Mp
for i=1:i_mouth-1 
  h_temp = h_mouth_shoal + slope_sea * (x_r(i_mouth,j)-x_r(i,j));
  h(i,j) = max( h(i,j), h_temp);
end
end
%%
h(find(h>depth_max)) = depth_max;

disp(['i_mouth   = ' num2str(i_mouth)])
disp(['j_river2  = ' num2str(j_river1)])

end

disp(['max depth is ' num2str(max(max(h)))])

yy = y_r(1,:) - y_r(1,j_river1);

figure;
plot(yy,-h(i_mouth:end,:)); xlim([-1 1]*1e4);
hold on
line([1 1]*yy(j_1),[-h_channel_max -h_mouth_shoal]);
line([1 1]*yy(j_2),[-h_channel_max -h_mouth_shoal]);
xlabel('along-shore distance '); ylabel('m'); title('channel profile');
%%
% ----------------------------------------------------------------------------

dx=dx_r'*ones(1,Mp);
dy=ones(Lp,1)*dy_r;
hbar=sum(sum(h.*dx.*dy))./sum(sum(dx.*dy));
disp(['Hbar = ',num2str(hbar)]);
disp(['Along channel gravity wave time = ',...
	num2str((xl./sqrt(9.8.*hbar))/86400,'%5.8f'),' days phase shift']);
disp(['Western cross-sectional area = ',...
	num2str(sum(h(2:end-1,1).*dy(2:end-1,1)))]);
disp(['Eastern cross-sectional area = ',...
	num2str(sum(h(2:end-1,end).*dy(2:end-1,end)))]);
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create masking grids, Coreolis grid, and angle.

% Masking at RHO-points.
mask_rho=ones(size(x_r));

if 1

for i=i_mouth:Lp
for j=1:(j_1-1)
  mask_rho(i,j)=0;
end
for j=(j_2+1):Mp
  mask_rho(i,j)=0;
end
end
end

 mask_rho(Lp,:)=0; 
% mask_rho(Lp-1,:)=0;
%%
for i=2:Lp,
  for j=1:Mp,
    umask(i-1,j)=mask_rho(i,j)*mask_rho(i-1,j);
  end,
end,
for i=1:Lp,
  for j=2:Mp,
    vmask(i,j-1)=mask_rho(i,j)*mask_rho(i,j-1);
  end,
end,
% vmask(Lp,:) = 0;
% vmask(Lp-1  ,:) = 0;

for i=2:Lp,
  for j=2:Mp,
    pmask(i-1,j-1)=mask_rho(i,j)*mask_rho(i-1,j)*mask_rho(i,j-1)*mask_rho(i-1,j-1);
  end,
end,

% Coriolis parameter.
rho.f=1e-4*ones(Lp,Mp) ;

% Angle of the grid is zero.
angle=zeros(size(x_r));
%%
rho.depth = h;
rho.x = x_r;
rho.y = y_r;
rho.mask = mask_rho;
save mat_grid_CASE1 rho projection spherical

  eval(['mat2roms_mw(''mat_grid_CASE1.mat'',''',fname,''');'])
  !del temp_jcw33.mat
  disp(['Created roms grid -->   ',fname])
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Create NetCDF file.
% 
% % Open file
% nc=netcdf(filename,'clobber');
% nc.Description = 'Estuary-Shelf Grid';
% nc.Author = 'Jia-Xuan Chang';
% nc.Created = datestr(now);
% nc.type = 'ESTUARY-SHELF GRID file';
% 
% % Dimensions
% nc('xi_rho')=Lp;
% nc('xi_u')  =L;
% nc('xi_v')  =Lp;
% nc('xi_psi')=L;
% 
% nc('eta_rho')=Mp;
% nc('eta_u')  =Mp;
% nc('eta_v')  =M;
% nc('eta_psi')=M;
% 
% nc('one')=1;
% 
% % Create variables
% dims = {'eta_rho'; 'xi_rho'};
% nc{'x_rho'}= ncdouble(dims);
% nc{'x_rho'}(:,:)=x_r;
% 
% dims = {'eta_psi'; 'xi_psi'};
% nc{'x_psi'}= ncdouble(dims);
% nc{'x_psi'}(:,:)=x_p;
% 
% dims = {'eta_u'; 'xi_u'};
% nc{'x_u'}= ncdouble(dims);
% nc{'x_u'}(:,:)=x_u;
% 
% dims = {'eta_v'; 'xi_v'};
% nc{'x_v'}= ncdouble(dims);
% nc{'x_v'}(:,:)=x_v;
% 
% dims = {'eta_rho'; 'xi_rho'};
% nc{'y_rho'}= ncdouble(dims);
% nc{'y_rho'}(:,:)=y_r;
% 
% dims = {'eta_psi'; 'xi_psi'};
% nc{'y_psi'}= ncdouble(dims);
% nc{'y_psi'}(:,:)=y_p;
% 
% dims = {'eta_u'; 'xi_u'};
% nc{'y_u'}= ncdouble(dims);
% nc{'y_u'}(:,:)=y_u;
% 
% dims = {'eta_v'; 'xi_v'};
% nc{'y_v'}= ncdouble(dims);
% nc{'y_v'}(:,:)=y_v;
% 
% dims = {'eta_rho'; 'xi_rho'};
% nc{'pm'}= ncdouble(dims);
% nc{'pm'}(:,:)=pm;
% 
% dims = {'eta_rho'; 'xi_rho'};
% nc{'pn'}= ncdouble(dims);
% nc{'pn'}(:,:)=pn;
% 
% dims = {'eta_rho'; 'xi_rho'};
% nc{'dmde'}= ncdouble(dims);
% nc{'dmde'}(:,:)=dmde;
% 
% dims = {'eta_rho'; 'xi_rho'};
% nc{'dndx'}= ncdouble(dims);
% nc{'dndx'}(:,:)=dndx;
% 
% dims = {'eta_rho'; 'xi_rho'};
% nc{'angle'}= ncdouble(dims);
% nc{'angle'}(:,:)=angle;
% 
% dims = {'eta_rho'; 'xi_rho'};
% nc{'mask_rho'}= ncdouble(dims);
% nc{'mask_rho'}(:,:)=mask_rho;
% 
% dims = {'eta_psi'; 'xi_psi'};
% nc{'mask_psi'}= ncdouble(dims);
% nc{'mask_psi'}(:,:)=pmask;
% 
% dims = {'eta_u'; 'xi_u'};
% nc{'mask_u'}= ncdouble(dims);
% nc{'mask_u'}(:,:)=umask;
% 
% dims = {'eta_v'; 'xi_v'};
% nc{'mask_v'}= ncdouble(dims);
% nc{'mask_v'}(:,:)=vmask;
% 
% dims = {'eta_rho'; 'xi_rho'};
% nc{'h'}= ncdouble(dims);
% nc{'h'}(:,:)=h;
% 
% dims = {'eta_rho'; 'xi_rho'};
% nc{'f'}= ncdouble(dims);
% nc{'f'}(:,:)=f;
% 
% dims = {'one'};
% nc{'el'} = ncdouble(dims);
% nc{'el'}(:) = el;
% 
% dims = {'one'};
% nc{'xl'} = ncdouble(dims);
% nc{'xl'}(:) = xl;
% 
% dims = {'one'};
% nc{'spherical'} = ncchar(dims);
% nc{'spherical'}(:) = 'F';
% 
% close(nc);
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Plotting
% plot1=0;
% 
% if plot1==1
% figure;clf
% mm = mask_rho;
% mm(find(mask_rho==0)) = nan;
% pcolor (x_r(2:end,2:end),y_r(2:end,2:end),...
%         h(2:end,2:end).*mm(2:end,2:end))
% %set(gca,'dataaspectratio',[1 1 1])
% hc=colorbar('horiz');
% hcl=get(hc,'ylabel');
% set(hcl,'string','Depth (m)');
% xlabel ('xi distance (m)');
% ylabel ('eta distance (m)');
% title ('Topography');
% 
% figure;clf
% subplot(2,1,1);
% plot (y_r(2:end-1,:),h(2:end-1,:),'-o');
% subplot(2,1,2);
% plot(x_r(1,:)-x_r(1,i_mouth),-h,'linewidth',2); 
% title(['bathy  river mouth idx = ' num2str(i_mouth)]);
% ylim([-max(max(h)) 0]); xlim([x_r(1,1)-x_r(1,i_mouth) x_r(1,end)-x_r(1,i_mouth)]);
% 
% figure;clf
% subplot(2,1,1);
% plot (x_r./1000,dx_r./1000);hold on;
% plot (x_r./1000,dx_r./1000,'o');
% xlabel('Along channel distance (km)');
% ylabel('Along channel grid spacing (km)');
% subplot(2,1,2);
% plot (y_r./1000,dy_r./1000);hold on;
% plot (y_r./1000,dy_r./1000,'o');
% xlabel('Cross channel distance (km)');
% ylabel('Cross channel grid spacing (km)');
% 
% end
% 
