% script create_roms_rivers
%
%
% Create a netcdf file that contains river forcing data for ROMS.
% Forcing data consists of:
% 'river_Xposition'  -   'river runoff  XI-positions at RHO-points'
% 'river_Eposition'  -   'river runoff ETA-positions at RHO-points'
% 'river_direction'  -   'river runoff direction'
% 'river_Vshape'     -   'river runoff mass transport vertical profile'
% 'river_transport'  -   'river runoff mass transport'
% 'river_temp'       -   'river runoff potential temperature'
% 'river_salt'       -   'river runoff salinity'
% 'river_mud_'       -   'river runoff suspended sediment concentration'
% 'river_sand_'      -   'river runoff suspended sediment concentration'
% 'river_dye_'       -   'river runoff dye concentration'   % add by SNC 1/5/11 ASR
%
% In your project.h you should have named now cppdefs.h
% #define TS_PSOURCE
% #define UV_PSOURCE
% #undef  ANA_PSOURCE
% define RIVER_SEDIMENT  % ASR
% #define T_PASSIVE      % ASR
%
%
% This m file is set to force rivers for LAKE_SIGNELL for ocean2.2 release.
% Users can adapt this file to their own application.
%
% jcw 5-25-2005
% jcw 21March2014  to use native matlab netcdf.
% Modified A Sanchez-Rios 2019

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Begin user input section                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 0)  Add paths for netcdf .m files from COAWST folder (ASR)
    addpath /Users/alesanchez/MyROMS/COAWST/Tools/mfiles/mtools
%1) Enter name of netcdf forcing file to be created.
%   If it already exists it will be overwritten!!.
forc_file='rivers_tracer_ale_C2.nc';

% If your adding a grid file uncomment below and add name (ASR)
grid_file='grid_case2_finergrid01_ale.nc'  ;  %<-enter name of grid here ASR

%2) Enter times of river forcings data, in seconds.
%   This time needs to be consistent with model time (ie dstart and time_ref).
%   See *.in files for more detail.

river_time = [0:3600:40*86400];                % 15 days   ASR


grd =  get_roms_grid(grid_file);                % ASR
[MP,LP] = size(grd.h);                         % ASR

load('riverfile_ale.mat');                         % ASR

for ii = 1:length(rout)
    Qr_flow(ii,1:length(river_time)) = rout(ii).Q;       % discharge; add by SNC ASR
end

% This is also added from the original not sure what this is ASR Sept 2019
if length(rout) > 1
    ns2       = length( rout(2).Y );
    j_channel = rout(2).Y(ceil(ns2/2)) + 1;      % plus one because Y has been subtracted 1
else
    j_channel = 1;
end

% This was in the original file
    num_river_times=length(river_time);     % do not change this.

%3) Enter number of vertical sigma levels in model.
%   This will be same value as entered in mod_param.F
        
    N=30;

%4) Enter the values of theta_s, theta_b, and Tcline from your *.in file.
    theta_s = 5;  
    theta_b = 0.4;  
    Tcline =  5; 
    Vtransform = 2;  % Added by ASR
    Vstretching = 4; % Added by ASR

%5) Enter value of h, Lm, and Mm.
%   This info can come from a grid file or user supplied here.
%   
%   Are you entering a grid file name (1 = yes, 0 = no)? 
    get_grid = 1;    %<--- put a 1 or 0 here
    
    if (get_grid)
    % Name added at the begining
%
% Get some grid info, do not change this.
%     % I (ASR) did not changed this as same as the 
      netcdf_load(grid_file);
      [LP,MP]=size(h);
      Lm=LP-2;
      Mm=MP-2;
%
    else
      Lm=100;       %<--- else put size of grid here, from mod_param.F
      Mm=20;        %<--- else put size of grid here, from mod_param.F
      LP = Lm+2;    %don't change this.
      MP = Mm+2;    %don't change this.

      % enter depth, same as in ana_grid
      for j=1:MP
        for i=1:LP
          h(j,i)=18-16*(Mm-j)/(Mm-1);
        end
      end
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% calc some grid stuff here - do not change this.
% You go on to step 6.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   L  = Lm+1;
   M  = Mm+1;
   xi_psi  = L;
   xi_rho  = LP;
   xi_u    = L;
   xi_v    = LP;
   eta_psi = M;
   eta_rho = MP;
   eta_u   = MP;
   eta_v   = M;
%
% Don't change this either.  This is from set_scoord.F
% This info is calculated so you know the vertical spacings
% of the grid at startup.
% Go to step 6 now.
%
%
% This was the original code from source %%%%%%%%%%%%%%%%%%%%%%%%%%
%    hmin=0;
%    hc=min([hmin,Tcline]);
%    if (theta_s~=0.0)
%      cff1=1.0/sinh(theta_s);
%      cff2=0.5/tanh(0.5*theta_s);
%    end
%    sc_w(1)=-1.0;
%    Cs_w(1)=-1.0;
%    cff=1.0/N;
%    for k=1:N
%      sc_w(k+1)=cff*(k-N);
%      sc_r(k)=cff*((k-N)-0.5);
%      if (theta_s~=0)
%        Cs_w(k+1)=(1.0-theta_b)*cff1*sinh(theta_s*sc_w(k+1))+   ...
%                       theta_b*(cff2*tanh(theta_s*(sc_w(k+1)+0.5))-0.5);
%        Cs_r(k)  =(1.0-theta_b)*cff1*sinh(theta_s*sc_r(k))+   ...
%                       theta_b*(cff2*tanh(theta_s*(sc_r(k)+0.5))-0.5);
%      else
%        Cs_w(k+1)=sc_w(k+1);
%        Cs_r(k)=sc_r(k);
%      end
%   end
%
% Now Added by SNC 2010  %%%%%%%%%%%%%%%%%%%%%%%%%%
   hmin = min(min(h));    % add by SNC 7/29/10
   hc = Tcline;
   [sc_r,Cs_r] = stretching(Vstretching,theta_s,theta_b,hc,N,0,0);
   [sc_w,Cs_w] = stretching(Vstretching,theta_s,theta_b,hc,N,1,0);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Assume zeta starts at 0.
%
    for j=1:eta_rho
      for i=1:xi_rho
        zeta(i,j,1) = 0;
      end
    end

    %%%%%% NEW from COD  folders
    %  Calc z at rho and w points.
%  Don't change this.
%
   [z_r] = set_depth(Vtransform,Vstretching,theta_s,theta_b,hc,N,1,h,squeeze(zeta(:,:,1)));
   [z_w] = set_depth(Vtransform,Vstretching,theta_s,theta_b,hc,N,5,h,squeeze(zeta(:,:,1)));
   z_r = permute(z_r,[1 2 3]);  %ASR change the order
   z_w = permute(z_w,[1 2 3]);  %ASR change the order
   Hz = diff(z_w,1,3);          %ASR change dimension
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% I think this is different in Shin-nans because we imposed the positions
% in the .mat file (ASR)
%6) Enter number of rivers.
%    num_rivers=6;
    numr = length(rout); 
    num_rivers = numr;

%7) Initialize river location and direction parameters.
%   Currently, the direction can be along XI-direction
%   (river-direction = 0) or along ETA-direction (river_direction > 0).  The mass sources are
%   located at U- or V-points so the grid locations should range from
%   1 =< river_Xposition =< L  and  1 =< river_Eposition =< M
% 
%     river_Xposition=[  31   56    7   64    4   8  77];   % num_rivers
%     river_Eposition=[1758 1621 1557 1545 1462 1418 508];  % num_rivers
%     river_direction=[0 0 0 0 0 0 0];      % num_rivers values
    river_Xposition = rout(1).X(:);  % ASR
    river_Eposition = rout(1).Y(:);  % ASR
    river_direction = rout(1).D(:);  % ASR
    
    mask_rho = ncread(grid_file,'mask_rho');  % ASR
    
    % Added in beta plume case  % ASR
      sig             = rout(1).sig(:);
    numcells        = length(rout(1).X) * ones(length(rout(1).X),1);    % number of cells for each river
    river_ID        = ones(length(rout(1).X),1);
    river_flag      = rout(1).flag(:);
    
    % In the cases of more that one river  % ASR
    if(numr > 1)
    for j = 2:numr
    river_Xposition = [river_Xposition; rout(j).X(:)];
    river_Eposition = [river_Eposition; rout(j).Y(:)];
    river_direction = [river_direction; rout(j).D(:)];
    sig             = [sig; rout(j).sig(:)];
    numcells        = [numcells; length(rout(j).X)*ones(length(rout(j).X),1)];
    river_ID        = [river_ID; j * ones(length(rout(j).X),1)];
    river_flag      = [river_flag; rout(j).flag(:)];
    end
    end

      Nsources = length(river_Xposition);  % ASR

%8) Initialize river shape.
    for i=1:Nsources
      for k=1:N
        river_Vshape(i,k)=1/N;
      end
    end

%% added figure ASR
    if 1
figure;
plot(Hz(:,j_channel,end),z_r(:,j_channel,end),'b-o'); hold on
%plot(Hz(k_temp:end,j_channel,end),z_r(k_temp:end,j_channel,end),'r'); 
    end
%%

%9) Initialize river flow.
%     river_transport=river_flow; %read in from file
%       river_transport=ones(num_rivers,num_river_times);
%%%%%%%%%% Added for Bplume case, this seems to set up when the river discharge starts 
% ASR
    ramp_u=0.5*24;              % start ramp UP at RAMP_UP hours
    ramp_time=1.0*24;         % ramp from 0 to 1 over RAMP_TIME hours
     ramp_time_2 = 1.0*24;
    ramp_d=60*24;              % start ramp DOWN at RAMP_DOWN hours
    fac=min( (0.5*(tanh((river_time/3600-ramp_u)/(ramp_time/5))+1)), ...
             (1-(0.5*(tanh((river_time/3600-ramp_d)/(ramp_time_2/5))+1))) );


    for i=1:Nsources
        Q_temp = Qr_flow(river_ID(i),:)/numcells(i);
         river_transport(:,i)= sig(i) .* fac .* Q_temp;
%         river_transport(:,i)= sig(i) .* Q_temp; % Trying something ASR
    end
%%%%%%%%%%%%%%%%%%%

%10) Time series of river temp and salt.
    for time=1:num_river_times
      for k=1:N
        for i=1:Nsources
          river_temp(i,k,time)=10;  % this is backwards in bplume
          river_salt(i,k,time)=rout(river_ID(i)).salt;
        end
      end
    end
    
   
%11) Enter number of mud sediments (NCS) and number of sand sediments (NNS).
%   These values should be the same as in mod_param.F
    NCS = 1;   %number of cohesive sed classes
    NNS = 0;   %number of non-cohesive sed classes
    NPT = 1;   %number of inactive passive tracer
%
% calc sed parameters. Do not alter.
%
   NAT=2;  %assume temp + salt are active
   NST = NCS + NNS;     % total number of sed tracers.
%    NT = NAT+NST;        % total number of tracers.
   NT = NAT+NST+NPT;        % total number of tracers. % ASR

%12) Sediment class properties (in order, mud first then sand).
%  These values should coincide with your sediment.in file.
  mud_Srho=ones(1,NCS)*2650;        %kg m-3, NCS values
  mud_Sd50=[0.03 0.03]/1000;        %m,      NCS values
  mud_Wsed=[0.0 0.0]/1000;         %m s-1,  NCS values
  mud_tau_ce=[0.05 0.05];           %N m-2,  NCS values
  mud_Erate=[5 5]*1e-5;             %kg m-2 s-1, NCS values
  sand_Srho=ones(1,NNS)*2650;       %kg m-3, NNS values
  sand_Sd50=[1.0]/1000;             %m,      NNS values
  sand_Wsed=[1.0]/1000;             %m s-1,  NNS values
  sand_tau_ce=[0.07];               %N m-2,  NNS values
  sand_Erate=[1]*1e-5;              %kg m-2 s-1, NNS values
%
% make some combined arrays.  Do not alter.
%
  Srho=  [mud_Srho,sand_Srho];
  Sd50=  [mud_Sd50,sand_Sd50];
  Wsed=  [mud_Wsed,sand_Wsed];
  tau_ce=[mud_tau_ce,sand_tau_ce];
  Erate= [mud_Erate,sand_Erate];


%13) Time series of river mud and sand.
%
% mud.
  display('Initializing river sediments.')

  % ASR
  b = 1;        % from sediment rating curve c = a * Q^b (see Kao & Milliman 2008)
  for idmud=1:NCS
    count=['0',num2str(idmud)];
    count=count(end-1:end);
    for time=1:num_river_times
      for k=1:N
        for i=1:Nsources
%           eval(['river_mud_',count,'(i,k,time) = 0;'])               %mud conc in water column
%           river_mud_temp(i,k,time) = C_river_temp(time);   % mud conc in river
          river_mud_temp(i,k,time) = 0;   % mud conc in river
        end
      end
    end
    eval(['river_mud_',count,' = river_mud_temp;'])             
    clear river_mud_temp; 
  end
%
% sand.
%
  for isand=1:NNS
    count=['0',num2str(isand)];
    count=count(end-1:end);
    for time=1:num_river_times
      for k=1:N
        for i=1:Nsources
%           eval(['river_sand_',count,'(i,k,time) = 0;'])               %sand conc in water column
       river_sand_temp(i,k,time) = 0;   % sand conc in river
        end
      end
    end
    eval(['river_sand_',count,' = river_sand_temp;'])             
    clear river_sand_temp; 
  end
  
  %%% This is new from SNS dye %14) Time series of river dye.  %ASR
%
if NPT ~= 0
  display('Initializing river inactive passive tracers.')
  C_dye_temp = river_time * 0;
%  C_dye_temp(find(river_time/86400 >= 1.5)) = C_dye;
%  C_dye_temp(find(river_time/86400 >= 0.0)) = C_dye;   % input dye at t=0

  for idpt=1:NPT
    count=['0',num2str(idpt)];
    count=count(end-1:end);
    for time=1:num_river_times
      for k=1:N
        for i=1:Nsources
          river_dye_temp(i,k,time) = rout(river_ID(i)).dye;   % dye conc in river
        end
      end
    end
    eval(['river_dye_',count,' = river_dye_temp;'])             
    clear river_dye_temp; 
  end
end

% Change to amake time in days now, as the code suggests.
river_time = river_time./86400;

%%%%% plotting some initial fields

figure;
subplot(2,2,1)
plot(river_time,sum(river_transport,2));
xlabel('time'); title('discharge');
subplot(2,2,2)
plot(river_time,squeeze(river_salt(:,end,:)));
xlabel('time'); title('salinity');
subplot(2,2,3)
plot(river_time,squeeze(river_mud_01(:,end,:)));
xlabel('time'); title('sed. conc');
subplot(2,2,4)
plot(river_time,squeeze(river_dye_01(:,end,:)));
xlabel('time'); title('dye conc');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  END of USER INPUT                                                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%create init file
nc_forc=netcdf.create(forc_file, 'clobber');
if isempty(nc_forc)
	disp([' ## Unable to create ROMS Rivers NetCDF file.'])
	return
end
 
%% Global attributes:
disp(' ## Defining Global Attributes...')
netcdf.putAtt(nc_forc,netcdf.getConstant('NC_GLOBAL'),'type','ROMS Rivers forcing file from create_roms_init.m');
netcdf.putAtt(nc_forc,netcdf.getConstant('NC_GLOBAL'),'history',['Created by ', mfilename ', on ', datestr(now)]);
netcdf.putAtt(nc_forc,netcdf.getConstant('NC_GLOBAL'),'title','ROMS Application')

%% Dimensions:
disp(' ## Defining Dimensions...')

% name of labels adjusted to fit the old name labels
xpsidimID = netcdf.defDim(nc_forc,'xi_psi',L);
xrhodimID = netcdf.defDim(nc_forc,'xi_rho',LP);
xudimID   = netcdf.defDim(nc_forc,'xi_u',L);
xvdimID   = netcdf.defDim(nc_forc,'xi_v',LP);

epsidimID = netcdf.defDim(nc_forc,'eta_psi',M);
erhodimID = netcdf.defDim(nc_forc,'eta_rho',MP);
eudimID   = netcdf.defDim(nc_forc,'eta_u',MP);
evdimID   = netcdf.defDim(nc_forc,'eta_v',M);

s_rhodimID = netcdf.defDim(nc_forc,'s_rho',N);
s_wdimID = netcdf.defDim(nc_forc,'s_w',N+1);
tracerdimID = netcdf.defDim(nc_forc,'tracer',NT);

riverdimID = netcdf.defDim(nc_forc,'river',Nsources)
river_timedimID = netcdf.defDim(nc_forc,'river_time',num_river_times)
onedimID = netcdf.defDim(nc_forc,'one',1);
twodimID = netcdf.defDim(nc_forc,'two',2);

%% Variables and attributes:
disp(' ## Defining Dimensions, Variables, and Attributes...')
 
theta_sID = netcdf.defVar(nc_forc,'theta_s','double',onedimID);
netcdf.putAtt(nc_forc,theta_sID,'long_name','S-coordinate surface control parameter');
netcdf.putAtt(nc_forc,theta_sID,'units','1');

theta_bID = netcdf.defVar(nc_forc,'theta_b','double',onedimID);
netcdf.putAtt(nc_forc,theta_bID,'long_name','S-coordinate bottom control parameter');
netcdf.putAtt(nc_forc,theta_bID,'units','1');

tcline_ID = netcdf.defVar(nc_forc,'Tcline','double',onedimID);
netcdf.putAtt(nc_forc,tcline_ID,'long_name','S-coordinate surface/bottom layer width');
netcdf.putAtt(nc_forc,tcline_ID,'units','meter');

hc_ID = netcdf.defVar(nc_forc,'hc','double',onedimID);
netcdf.putAtt(nc_forc,hc_ID,'long_name','S-coordinate parameter, critical depth');
netcdf.putAtt(nc_forc,hc_ID,'units','meter');

Cs_rID = netcdf.defVar(nc_forc,'Cs_r','double',s_rhodimID);
netcdf.putAtt(nc_forc,Cs_rID,'long_name','S-coordinate stretching curves at RHO-points');
netcdf.putAtt(nc_forc,Cs_rID,'units','1');
netcdf.putAtt(nc_forc,Cs_rID,'valid_min',-1);
netcdf.putAtt(nc_forc,Cs_rID,'valid_max',0);
netcdf.putAtt(nc_forc,Cs_rID,'field','Cs_r, scalar');

Cs_wID = netcdf.defVar(nc_forc,'Cs_w','double',s_wdimID);
netcdf.putAtt(nc_forc,Cs_wID,'long_name','S-coordinate stretching curves at W-points');
netcdf.putAtt(nc_forc,Cs_wID,'units','1');
netcdf.putAtt(nc_forc,Cs_wID,'valid_min',-1);
netcdf.putAtt(nc_forc,Cs_wID,'valid_max',0);
netcdf.putAtt(nc_forc,Cs_wID,'field','Cs_w, scalar');

sc_rID = netcdf.defVar(nc_forc,'sc_r','double',s_rhodimID);
netcdf.putAtt(nc_forc,sc_rID,'long_name','S-coordinate at RHO-points');
netcdf.putAtt(nc_forc,sc_rID,'units','1');
netcdf.putAtt(nc_forc,sc_rID,'valid_min',-1);
netcdf.putAtt(nc_forc,sc_rID,'valid_max',0);
netcdf.putAtt(nc_forc,sc_rID,'field','sc_r, scalar');

sc_wID = netcdf.defVar(nc_forc,'sc_w','double',s_wdimID);
netcdf.putAtt(nc_forc,sc_wID,'long_name','S-coordinate at W-points');
netcdf.putAtt(nc_forc,sc_wID,'units','1');
netcdf.putAtt(nc_forc,sc_wID,'valid_min',-1);
netcdf.putAtt(nc_forc,sc_wID,'valid_max',0);
netcdf.putAtt(nc_forc,sc_wID,'field','sc_w, scalar');

river_ID = netcdf.defVar(nc_forc,'river','double',riverdimID);
netcdf.putAtt(nc_forc,river_ID,'long_name','river_runoff identification number');
netcdf.putAtt(nc_forc,river_ID,'units','nondimensional');
netcdf.putAtt(nc_forc,river_ID,'field','num_rivers, scalar, series');

river_timeID = netcdf.defVar(nc_forc,'river_time','double',river_timedimID);
netcdf.putAtt(nc_forc,river_timeID,'long_name','river time');
netcdf.putAtt(nc_forc,river_timeID,'units','days');
netcdf.putAtt(nc_forc,river_timeID,'field','river_time, scalar, series');

river_XpositionID = netcdf.defVar(nc_forc,'river_Xposition','double',riverdimID);
netcdf.putAtt(nc_forc,river_XpositionID,'long_name','river runoff  XI-positions at RHO-points');
netcdf.putAtt(nc_forc,river_XpositionID,'units','scalar');
netcdf.putAtt(nc_forc,river_XpositionID,'time','river_time');
netcdf.putAtt(nc_forc,river_XpositionID,'field','river runoff XI position, scalar, series');

river_EpositionID = netcdf.defVar(nc_forc,'river_Eposition','double',riverdimID);
netcdf.putAtt(nc_forc,river_EpositionID,'long_name','river runoff  ETA-positions at RHO-points');
netcdf.putAtt(nc_forc,river_EpositionID,'units','scalar');
netcdf.putAtt(nc_forc,river_EpositionID,'time','river_time');
netcdf.putAtt(nc_forc,river_EpositionID,'field','river runoff ETA position, scalar, series');

river_directionID = netcdf.defVar(nc_forc,'river_direction','double',riverdimID);
netcdf.putAtt(nc_forc,river_directionID,'long_name','river runoff direction, XI=0, ETA>0');
netcdf.putAtt(nc_forc,river_directionID,'units','scalar');
netcdf.putAtt(nc_forc,river_directionID,'time','river_time');
netcdf.putAtt(nc_forc,river_directionID,'field','river runoff direction, scalar, series');

river_VshapeID = netcdf.defVar(nc_forc,'river_Vshape','double',[riverdimID s_rhodimID]);
netcdf.putAtt(nc_forc,river_VshapeID,'long_name','river runoff mass transport vertical profile');
netcdf.putAtt(nc_forc,river_VshapeID,'units','scalar');
netcdf.putAtt(nc_forc,river_VshapeID,'time','river_time');
netcdf.putAtt(nc_forc,river_VshapeID,'field','river runoff vertical profile, scalar, series');

river_transportID = netcdf.defVar(nc_forc,'river_transport','double',[riverdimID river_timedimID]);
netcdf.putAtt(nc_forc,river_transportID,'long_name','river runoff mass transport');
netcdf.putAtt(nc_forc,river_transportID,'units','meter^3/s');
netcdf.putAtt(nc_forc,river_transportID,'time','river_time');
netcdf.putAtt(nc_forc,river_transportID,'field','river runoff mass transport, scalar, series');

river_saltID = netcdf.defVar(nc_forc,'river_flag','double',[riverdimID]);
netcdf.putAtt(nc_forc,river_saltID,'long_name','river flag, 1=temp, 2=salt, 3=temp+salt, 4=temp+salt+sed, 5=temp+salt+sed+bio');
netcdf.putAtt(nc_forc,river_saltID,'units','nondimension');
netcdf.putAtt(nc_forc,river_saltID,'time','river_time');
netcdf.putAtt(nc_forc,river_saltID,'field','river flag, scalar, series');

river_tempID = netcdf.defVar(nc_forc,'river_temp','double',[riverdimID s_rhodimID river_timedimID]);
netcdf.putAtt(nc_forc,river_tempID,'long_name','river runoff potential temperature');
netcdf.putAtt(nc_forc,river_tempID,'units','Celsius');
netcdf.putAtt(nc_forc,river_tempID,'time','river_time');
netcdf.putAtt(nc_forc,river_tempID,'field','river temperature, scalar, series');

river_saltID = netcdf.defVar(nc_forc,'river_salt','double',[riverdimID s_rhodimID river_timedimID]);
netcdf.putAtt(nc_forc,river_saltID,'long_name','river runoff salinity');
netcdf.putAtt(nc_forc,river_saltID,'units','PSU');
netcdf.putAtt(nc_forc,river_saltID,'time','river_time');
netcdf.putAtt(nc_forc,river_saltID,'field','river salinity, scalar, series');

% new Ale
if NPT ~=0
for idpt=1:NPT
  count=['00',num2str(idpt)];
  count=count(end-1:end);
  eval(['river_dye_',count,'ID = netcdf.defVar(nc_forc,''river_dye_',count,''',''double'',[riverdimID s_rhodimID river_timedimID]);'])
  eval(['netcdf.putAtt(nc_forc,river_dye_',count,'ID,''long_name'',''river runoff dye concentration, dye ',count,''');'])
  eval(['netcdf.putAtt(nc_forc,river_dye_',count,'ID,''units'',''kilogram meter-3'');'])
  eval(['netcdf.putAtt(nc_forc,river_dye_',count,'ID,''time'',''river_time'');'])
  eval(['netcdf.putAtt(nc_forc,river_dye_',count,'ID,''field'',''river runoff dye_',count,', scalar, series'');'])    
end
end
 
for mm=1:NCS
  count=['00',num2str(mm)];
  count=count(end-1:end);
  eval(['river_mud_',count,'ID = netcdf.defVar(nc_forc,''river_mud_',count,''',''double'',[riverdimID s_rhodimID river_timedimID]);'])
  eval(['netcdf.putAtt(nc_forc,river_mud_',count,'ID,''long_name'',''river runoff suspended sediment concentration, size class ',count,''');'])
  eval(['netcdf.putAtt(nc_forc,river_mud_',count,'ID,''units'',''kilogram meter-3'');'])
  eval(['netcdf.putAtt(nc_forc,river_mud_',count,'ID,''time'',''river_time'');'])
  eval(['netcdf.putAtt(nc_forc,river_mud_',count,'ID,''field'',''river runoff mud_',count,', scalar, series'');'])    
end
for mm=1:NNS
  count=['00',num2str(mm)];
  count=count(end-1:end);
  eval(['river_sand_',count,'ID = netcdf.defVar(nc_forc,''river_sand_',count,''',''double'',[riverdimID s_rhodimID river_timedimID]);'])
  eval(['netcdf.putAtt(nc_forc,river_sand_',count,'ID,''long_name'',''river runoff suspended sediment concentration, size class ',count,''');'])
  eval(['netcdf.putAtt(nc_forc,river_sand_',count,'ID,''units'',''kilogram meter-3'');'])
  eval(['netcdf.putAtt(nc_forc,river_sand_',count,'ID,''time'',''river_time'');'])
  eval(['netcdf.putAtt(nc_forc,river_sand_',count,'ID,''field'',''river runoff sand_',count,', scalar, series'');'])    
end
netcdf.close(nc_forc)

%now write the data from the arrays to the netcdf file
disp(' ## Filling Variables in netcdf file with data...')

ncwrite(forc_file,'theta_s',theta_s);
ncwrite(forc_file,'theta_b',theta_b);
ncwrite(forc_file,'Tcline',Tcline);
ncwrite(forc_file,'Cs_r',Cs_r);
ncwrite(forc_file,'Cs_w',Cs_w);
ncwrite(forc_file,'sc_w',sc_w);
ncwrite(forc_file,'sc_r',sc_r);
ncwrite(forc_file,'hc',hc);
ncwrite(forc_file,'river',[1:num_rivers]);

ncwrite(forc_file,'river_time',river_time);
ncwrite(forc_file,'river_Xposition',river_Xposition);
ncwrite(forc_file,'river_Eposition',river_Eposition);
ncwrite(forc_file,'river_direction',river_direction);
ncwrite(forc_file,'river_Vshape',river_Vshape);
ncwrite(forc_file,'river_transport',river_transport');
ncwrite(forc_file,'river_temp',river_temp);
ncwrite(forc_file,'river_salt',river_salt);

if NPT ~= 0
for mm=1:NPT
  count=['00',num2str(mm)];
  count=count(end-1:end);
  eval(['ncwrite(forc_file,''river_dye_',count,''',river_dye_',count,');'])    %dy conc in water column
end
end

for mm=1:NCS
  count=['00',num2str(mm)];
  count=count(end-1:end);
  eval(['ncwrite(forc_file,''river_mud_',count,''',river_mud_',count,');'])    %mud conc in water column
end
for mm=1:NNS
  count=['00',num2str(mm)];
  count=count(end-1:end);
  eval(['ncwrite(forc_file,''river_sand_',count,''',river_sand_',count,');'])  %sand conc in water column
end

%close file
disp(['created ', forc_file])


