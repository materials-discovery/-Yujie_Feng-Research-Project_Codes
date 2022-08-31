%  Simple Continuous GA
% minimizes the objective function designated in ff
% Before beginning, set all the parameters in parts
% I, II, and III
% Haupt & Haupt
% Adrian Rubio-Solis
% Institute for Materials Discovery (UCL)
% 2003
%__________________________________________________________________________
% Reading the Data Set
clc;
clear;

% rand('seed',1)
data_set = xlsread('/Users/jade/Library/CloudStorage/OneDrive-UniversityCollegeLondon/School/Research Project/Dataset_4.xlsx');

input_data = data_set(:,2:12); % this is to copy the columns between 2-14 which are the inputs
target = data_set(:,16);

% Normalization of input data set
X = normalisation_1(input_data, 2);

 %+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% min_value = min(target);
% max_value = max(target);
%--------------------------------------------------------------------------
%                       CROSS-VALIDATION DATA SPLIT
% Spliting the data into two subsets: Training and testing
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%                       CROSS-VALIDATION DATA SPLIT
%--------------------------------------------------------------------------
samples4training = round(0.8*size(X,1));
dimension = size(X,2);
samples4testing = size(X,1) - (samples4training);
px = randperm(size(X,1));
% Variables for training
data4training = zeros(samples4training,dimension);
label4training = zeros(samples4training,1);
% Variables for testing
data4testing = zeros(samples4testing,dimension);
label4testing = zeros(samples4testing,1);
% %..........................................................................
% %% Data 4 Training
% %..........................................................................
% for training (data for)
index = 0;

   for k = 1:samples4training
         index = index + 1;
         data4training(index,:) = X(px(k),1:end);
         label4training(index,1) = target(px(k),1);
   end
%           Creating Testing Data
index = 0;
%..........................................................................
% this is the loopr testing
    for k = samples4training + 1:samples4training + samples4testing 
        index = index + 1;
                data4testing(index,:) = X(px(k),1:end);
                label4testing(index,1) = target(px(k),1);
    end
    
%..........................................................................s
%__________________________________________________________________________
% I Setup the GA  Parameters
NumberofHiddenNeurons = 30;
npar = NumberofHiddenNeurons*dimension + NumberofHiddenNeurons; % number of optimization variables 
varhi=0.5; % high limit
varlo=-0.1; % lower limit
%__________________________________________________________________________
maxit = 100;                      % Max number of iterations or max number of generations
best_fitness = zeros(maxit,1);
mincost=-9999999;
                            %II Stopping criteria
                   % max number of iterations % minimum cost
%__________________________________________________________________________
%                           III GA parameters 
popsize = 30;   % set population size   (only pair numbers) 
mutrate = 0.005; % [0-1]  % 10% the reference set mutation rate
selection = 0.7;% [0.4-0.70] % 450% or the number of individuals that survivemore than one generation
%                fraction of population kept
Nt = npar; % continuous parameter GA Nt=#variables

keep = floor(selection*popsize); % #population (number of individuals)
                                 % number of members that survive
                                 % members that survive
nmut = ceil((popsize-1)*Nt*mutrate); % total number of % mutations
M = ceil((popsize-keep)/2);          % number of matings
%__________________________________________________________________________ 
%                   Create the initial population
%__________________________________________________________________________
generation = 0; % generation counter initialized
par=(varhi-varlo)*rand(popsize,npar)+varlo;     % random generation of parameters
% evaluating initial cost (fitness) of the initial population
InputWeight = zeros(NumberofHiddenNeurons,dimension);
NumberofTrainingData = samples4training;
ActivationFunction = 'Sine';        % Select the type of hidden neuron model
% List of models for the hidden neurons
% sine, hardlim, tribas, radbas
%__________________________________________________________________________
for i = 1:popsize  
    % Convert chromosome into a matrix format
    % par is the variable used to define the crhomosone of each individual
    [ c(i).OutputWeight, Y, rmse ] = ELM_model( par(i,:), NumberofHiddenNeurons, data4training, label4training,ActivationFunction);
    cost(i,1) = rmse;
end                    
[cost,ind] = sort(cost);          % min cost in element 1
par = par(ind,:);               % sort continuous
minc(1) = min(cost);            % minc contains min of
meanc(1)= mean(cost);           % meanc contains mean of population
%__________________________________________________________________________ 
                        % Iterate through generations
rmse_training = zeros(maxit,1);                        
%  HERE ids where the loop for iterating through generations STARTS                        
while generation<maxit
 
 generation = generation+1;             % increments generation counter
%.......................................................................... 
% %                      - Pair and mate - 
%..........................................................................
 M = ceil((popsize-keep)/2);                    % number of matings
 % Roulette WHEEL
 prob = flipud([1:keep]'/sum([1:keep]));        % weights
%                                               % chromosomes
 odds=[0 cumsum(prob(1:keep))'];                % probability      
%                                               % probability
%                                               % distribution
 pick1=rand(1,M);                               %  mate #1
 pick2=rand(1,M);                               % mate #2
% % ma and pa contain the indicies of the chromosomes 
% % that will mate
 ic=1;
 % loop for the number of matings
 while ic<=M
  % Pick any two individuals based on the probability of the roulette wheel
  for id=2:keep+1
     if pick1(ic)<=odds(id) & pick1(ic)>odds(id-1)
         ma(ic)=id-1;
     end
     if pick2(ic)<=odds(id) & pick2(ic)>odds(id-1) 
         pa(ic)=id-1;
     end
   end
    ic=ic+1; 
 end
%__________________________________________________________________________
%..........................................................................
%    - Performs mating using single point crossover -   MATING
%..........................................................................
 ix = 1:2:keep;                            % index of mate 1
 xp = ceil(rand(1,M)*Nt);                  % cross over point 
 r = rand(1,M);                            % mixing parameter
 for ic=1:M
     xy = par( ma(ic),xp(ic) ) - par(pa(ic),xp(ic)); % ma and pa    
%                                               % mate
     par(keep+ix(ic),:)   =   par(ma(ic),:); % 1st offspring 
     par(keep+ix(ic)+1,:) =   par(pa(ic),:); % 2nd offspring 
% % % 1st offspring
  par(keep+ix(ic),xp(ic))= par(ma(ic),xp(ic)) - r(ic).*xy;
% % % 2nd offspring
  par(keep+ix(ic)+1,xp(ic))=par( pa(ic),xp(ic) ) + r(ic).*xy; 
  if xp(ic)<npar % crossover when last variable not selected
      par(keep+ix(ic),:)= [par(keep+ix(ic),1:xp(ic)) par(keep+ix(ic)+1,xp(ic)+1:npar)];
      par(keep+ix(ic)+1,:)= [par(keep+ix(ic)+1,1:xp(ic)) par(keep+ix(ic),xp(ic)+1:npar)];
  
   end % if 
 end
%__________________________________________________________________________ 
                           % Mutate the population 
 mrow=sort(ceil(rand(1,nmut)*(popsize-1))+1); 
 mcol=ceil(rand(1,nmut)*Nt);
  
  for ii=1:nmut 
      par(mrow(ii),mcol(ii))=(varhi-varlo)*rand+varlo;
                              % mutation 
  end % ii
%__________________________________________________________________________
% The new offspring and mutated chromosomes are % evaluated
%cost=feval(ff,par);
for i = 1:popsize
    %cost(i,1) = feval(ff, par(i,:));   
    % Convert chromosome into a matrix format
    [ c(i).OutputWeight,Y, rmse ] = ELM_model( par(i,:), NumberofHiddenNeurons, data4training, label4training,ActivationFunction);
    cost(i,1) = rmse;
end
%__________________________________________________________________________ 
 % Sort the costs and associated parameters 
 [cost,ind]=sort(cost);
 par=par(ind,:);
%__________________________________________________________________________ 
% Do statistics for a single nonaveraging run
    minc(generation+1) = min(cost);
    meanc(generation+1)= mean(cost);
%__________________________________________________________________________ 
                              % Stopping criteria
   if generation > maxit | cost(1)<mincost
       best_fitness(generation,1)=cost(1);
       break 
   else
       best_fitness(generation,1)=cost(1);
   end
   rmse_training(generation,1) = best_fitness(generation,1);
   [generation cost(1)] 
   % cost (1) -- best RMSE at each iteration of the training found so far
   
 end %number_of_generation
%__________________________________________________________________________ 
%%                       - Testing Results/stage-
%__________________________________________________________________________ 
% In this stage, the parameters und for the perceptron model are tested
% withew/unseen data created by spliting the originaata Such data is
% usually called testing data
%__________________________________________________________________________
% caklculate matrx H for testing data
    
  [ TY ] = model4testing( data4testing, NumberofHiddenNeurons, c(1).OutputWeight, par(ind(1),1:end), ActivationFunction );
% Computing Root-Mean-Square-Error (RMSE) for testing
  TV.T = label4testing';
  rmse_test=sqrt(mse(TV.T - TY));
  %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  % Training
  R2_train = rsquare(label4training,Y');
  % Testing
  R2_test = rsquare(label4testing,TY');
  [R2_test,rmse_test]

 % axis([0 25 0 25]) ;
 % hold on;
 % xlabel('Experimental PCE (%)','FontSize',16);
 % ylabel('Predicted PCE (%)','FontSize',16);
 % plot(label4testing,label4testing,'k');

 % scatter(label4testing,TY','MarkerFaceColor',[0 0.4470 0.7410],'MarkerEdgeColor',[0 0.4470 0.7410])

%..........................................................................
%                  - SECTION FOR PLOTTING THE 3D SURFACE -
%..........................................................................
 %%
% varX = 1;
% varY = 2;
% res = 40;
% [ X1, Y1, Z1 ] = Optimal_region_surface_Tc(varX, varY, X, res, NumberofHiddenNeurons, par(ind(1),1:end), c(1).OutputWeight, ActivationFunction, label4training );
%--------------------------------------------------------------------------

% denormalization
% raw_data4training = input_data;
% create X vector
% for jj=1:res
%    for ii=1:res % here the conversion from normalised to real data again is based on the fact you normalised between 0-1
%        X1(jj,ii)= min(raw_data4training(:,varX)) + ii*((max(raw_data4training(:,varX)) - min(raw_data4training(:,varX)))/res);
%    end
% end
% create Y vector
% for jj=1:res
%    for ii=1:res
%        Y1(ii,jj)= min(raw_data4training(:,varY))+  ii*((max(raw_data4training(:,varY)) - min(raw_data4training(:,varY)))/res);
%    end
% end


% Plotting Surface
% surf(X1,Y1,Z1,'FaceAlpha',0.5);
 
% zmax=max(max(Z1));
% [id_xmax,id_ymax]=find(Z1==zmax);
% xmax=X1(id_xmax);
% ymax=Y1(id_ymax);
% hold on;
% plot3(xmax,ymax,zmax,'k.','markersize',20);
% text(xmax,ymax,zmax,[' x=',num2str(xmax),char(10),' y=',num2str(ymax),char(10),' z=',num2str(zmax),char(10),]);


% xlabel('CGI');
% ylabel('GGI(F)');
% zlabel('Predicted PCE (%)');
% colorbar;


%%   Denormalization process
% denormalized_data4training = zeros(size(data4training,1),size(data4training,2));
% nose_point = 500;
% number_of_points = 20;
% lower_part = zeros(1,number_of_points);
% lower_times = zeros(1,number_of_points);
% inc_T = 500;
% Ti = nose_point;
%--------------------------------------------------------------------------
%%
%for i = 1:size(data4training,1)
%  for k = 1:size(data4training,2)
%     denormalized_data4training(i,k) = data4training(i,k)*(maximo_M(1,k)-minimo_M(1,k)) + minimo_M(1,k);  
%  end
%end
%--------------------------------------------------------------------------
%% sort is to arrange the data from the smallest value to the biggest one
% [ordered_times,nx] = sort(denormalized_data4training(:,end));
% ordered_predicted = Y(1,nx);
% ordered_measured = label4training(nx, 1);% this should be a row vector

% for k = 1:size()
% 
% end


