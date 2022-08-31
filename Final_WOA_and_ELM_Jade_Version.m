%_________________________________________________________________________%
%  Whale Optimization Algorithm (WOA) source codes demo 1.0               %
%                                                                         %
%  Developed in MATLAB R2011b(7.13)                                       %
%                                                                         %
%  Author and programmer: Seyedali Mirjalili                              %
%                                                                         %
%         e-Mail: ali.mirjalili@gmail.com                                 %
%                 seyedali.mirjalili@griffithuni.edu.au                   %
%                                                                         %
%       Homepage: http://www.alimirjalili.com                             %
%                                                                         %
%   Main paper: S. Mirjalili, A. Lewis                                    %
%               The Whale Optimization Algorithm,                         %
%               Advances in Engineering Software , in press,              %
%               DOI: http://dx.doi.org/10.1016/j.advengsoft.2016.01.008   %
%                                                                         %
%_________________________________________________________________________%

% You can simply define your cost in a seperate file and load its handle to fobj 
% The initial parameters that you need are:
%__________________________________________
% fobj = @YourCostFunction
% dim = number of your variables
% Max_iteration = maximum number of generations
% SearchAgents_no = number of search agents
% lb=[lb1,lb2,...,lbn] where lbn is the lower bound of variable n
% ub=[ub1,ub2,...,ubn] where ubn is the upper bound of variable n
% If all the variables have equal lower bound you can just
% define lb and ub as two single number numbers

% To run WOA: [Best_score,Best_pos,WOA_cg_curve]=WOA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj)
%__________________________________________

%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%%
clear;
clc;

% rand('seed',0)
% load processed_data.mat; % reading the data set/
filename = '/Users/jade/Library/CloudStorage/OneDrive-UniversityCollegeLondon/School/Research Project/Dataset_4.xlsx';

rawdata = xlsread(filename);
% Input Data to the Machine Learning

input_data = rawdata(:,2:12);
% Target
Y = rawdata(:,16);
% Y = xlsread(filename)

%[ X, maximo_M, minimo_M ] = normalisation_R( input_data, 2 );

X = normalisation_1(input_data, 2);

% min_value = min(Y);
% max_value = max(Y);

%--------------------------------------------------------------------------
%                       CROSS-VALIDATION DATA SPLIT
% Spliting the data into two subsets: Training and testing
%--------------------------------------------------------------------------
samples4training = round(0.8*size(X,1)); % size(X,1):row
dimension = size(X,2); % size(X,2):column
samples4testing = size(X,1) - (samples4training);

px = randperm(size(X,1)); % p = randperm(n) returns a row vector containing a random permutation of the integers from 1 to n without repeating elements.
% Variables for training
data4training = zeros(samples4training,dimension);  % this is the input matrix for training
label4training = zeros(samples4training,1);  % this is the target for training
% Variables for testing
data4testing = zeros(samples4testing,dimension); % this is the matrix for testing
label4testing = zeros(samples4testing,1); % this is the target for testing
%..........................................................................
%..........................................................................
%%                           Data for Training
%..........................................................................
index = 0; 

% PROPERTY_CIGS = 1; % PROPERTY_CIGS = 1= Jsc,PROPERTY_CIGS = 2 = Voc
%                     % PROPERTY_CIGS = 3 = FF, PROPERTY_CIGS = 4 = PCE
   for k = 1:samples4training 
         index = index + 1;
         data4training(index,:) = X(px(k),1:end); 
         label4training(index,1) = Y(px(k));
   end
%%           Creating Testing Data
%..........................................................................
index = 0;

    for k = samples4training + 1:samples4training + samples4testing 
        index = index + 1;
                data4testing(index,:) = X(px(k),1:end);
                label4testing(index,1) = Y(px(k));
    end
%..........................................................................
%% Parameters for the Perceptron Neural Network
NumberofHiddenNeurons = 30;
ActivationFunction = 'Sine';
T = label4training';
% List of models for the hidden neurons
% sine, hardlim, tribas, radbas
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameters for WOA
SearchAgents_no=40; % Number of search agents

%Function_name='F1'; % Name of the test function that can be from F1 to F23 (Table 1,2,3 in the paper)

Max_iter=100; % Maximum numbef of iterations

% Load details of the selected benchmark function
%[lb,ub,dim,fobj]=Get_Functions_details(Function_name);
dim = NumberofHiddenNeurons*dimension + NumberofHiddenNeurons; % number of parameters (number of input weights + number of biases)  
lb = -3; % lb = lower boundary
ub = 3; % ub = upper boundary
%..........................................................................
%..........................................................................
%[Best_score,Best_pos,WOA_cg_curve]=WOA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
%function [Leader_score,Leader_pos,Convergence_curve]=WOA(SearchAgents_no,Max_iter,lb,ub,dim,fobj)

% initialize position vector and score for the leader
Leader_pos=zeros(1,dim);
Leader_score=inf; %change this to -inf for maximization problems

%Initialize the positions of search agents
Positions=initialization(SearchAgents_no,dim,ub,lb);

Convergence_curve=zeros(1,Max_iter);

t = 0;% Loop counter
rmse_training = zeros(Max_iter,1);
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%                       - Main loop for iterations -
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
while t<Max_iter
    for i=1:size(Positions,1)
        
        % Return back the search agents that go beyond the boundaries of the search space
        Flag4ub=Positions(i,:)>ub;
        Flag4lb=Positions(i,:)<lb;
        Positions(i,:)=(Positions(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
        
        % Calculate objective function for each search agent
        %fitness=fobj(Positions(i,:));
        [ c(i).OutputWeight, Y, fitness ] = ELM_model( Positions(i,:), NumberofHiddenNeurons, data4training,label4training, ActivationFunction);
         
        % Y -- output of the neural network
        % fitness -- RMSE

        % Update the leader
        if fitness<Leader_score % Change this to > for maximization problem

            Leader_score = fitness; % Update alpha

            Leader_pos = Positions(i,:);

            Leader_OutputWeight = c(i).OutputWeight;
            
        end
        
    end
    
    a=2-t*((2)/Max_iter); % a decreases linearly fron 2 to 0 in Eq. (2.3)
    
    % a2 linearly dicreases from -1 to -2 to calculate t in Eq. (3.12)
    a2=-1+t*((-1)/Max_iter);
    
    % Update the Position of search agents 
    for i=1:size(Positions,1)
        
        r1=rand(); % r1 is a random number in [0,1]
        r2=rand(); % r2 is a random number in [0,1]

        
        A=2*a*r1-a;  % Eq. (2.3) in the paper
        C=2*r2;      % Eq. (2.4) in the paper
        
        
        b=1;               %  parameters in Eq. (2.5)
        l=(a2-1)*rand+1;   %  parameters in Eq. (2.5)，l属于【-1，1】
        
        p = rand();        % p in Eq. (2.6)
        
        for j=1:size(Positions,2)
            
            if p<0.5 % shrinking encircling mechanism  

                % If｜A｜> 1, choose a random search (go back to search for preu [exploration phase] ?)

                if abs(A)>=1 % Y = abs(X) 返回数组 X 中每个元素的绝对值
                    rand_leader_index = floor(SearchAgents_no*rand()+1);
                    X_rand = Positions(rand_leader_index, :);
                    D_X_rand=abs(C*X_rand(j)-Positions(i,j)); % Eq. (2.7)
                    Positions(i,j)=X_rand(j)-A*D_X_rand;      % Eq. (2.8)
                    
                % If｜A｜< 1, the best solution is selected (go to encircling prey ?)

                elseif abs(A)<1
                    D_Leader=abs(C*Leader_pos(j)-Positions(i,j)); % Eq. (2.1)
                    Positions(i,j)=Leader_pos(j)-A*D_Leader;      % Eq. (2.2)
                end
                

            elseif p>=0.5 % spiral updating position
              
                distance2Leader=abs(Leader_pos(j)-Positions(i,j));
                % Eq. (2.5)
                Positions(i,j)=distance2Leader*exp(b.*l).*cos(l.*2*pi)+Leader_pos(j);
                
            end
            
        end
    end
    t=t+1;
    rmse_training(t,1) = Leader_score;
    Convergence_curve(t)=Leader_score;
    [t Leader_score]
end
Best_score = Leader_score;
Best_pos = Leader_pos;
WOA_cg_curve = Convergence_curve;
%..........................................................................
% %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%__________________________________________________________________________ 
%%                       - Testing Results/stage-
%__________________________________________________________________________ 
% In this stage, the parameters und for the perceptron model are tested
% withew/unseen data created by spliting the originaata Such data is
% usually called testing data
%__________________________________________________________________________
% calculate matrx H for testing data
    
  [ TY ] = model4testing( data4testing, NumberofHiddenNeurons, Leader_OutputWeight , Leader_pos, ActivationFunction );
% Computing Root-Mean-Square-Error (RMSE) for testing
  TV.T = label4testing';
  rmse_test=sqrt(mse(TV.T - TY));  
  R2_test = rsquare(label4testing,TY');
  [R2_test,rmse_test]


 % axis([0 25 0 25]) ;
 % hold on;
 % xlabel('Experimental PCE (%)','FontSize',16);
 % ylabel('Predicted PCE (%)','FontSize',16);
 % plot(label4testing,label4testing,'k');
 % scatter(label4testing,TY','MarkerFaceColor',[0 0.4470 0.7410],'MarkerEdgeColor',[0 0.4470 0.7410])

  
  % long = -label4testing(:,2);   
  % lat = label4testing(:,3); 
  % zline = label4testing(:,4);

  % bar = TY'(:)

  % scatter3(long,lat,rural,40,zline,'filled')







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

%maximo_M = max(input_data)
%minimo_M = min(input_data)
%for i = 1:size(input_data,1)
%  for k = 1:size(input_data,2)
%     denormalized_input_data(i,k) = input_data(i,k)*(maximo_M(1,k)-minimo_M(1,k)) + minimo_M(1,k);  
% end
%end     



 % R2_train = rsquare(label4training,Y');
 % R2_test = rsquare(lable4testing,TY');
 % [R2_train R2_test]
%%
% __________________________________________________________________________
%..........................................................................
%                  - SECTION FOR PLOTTING THE 3D SURFACE -
%..........................................................................

% max_values = max(X);
% min_values = min(X);

 varX = 8; % where is it defined? -- in the "Optimal_region_surface.m" file
 varY = 8;
 res = 40;
 [ X1, Y1, Z1 ] = Optimal_region_surface( varX, varY, X, res, NumberofHiddenNeurons, Leader_pos(1,1:end), Leader_OutputWeight, ActivationFunction );
%--------------------------------------------------------------------------
  % Plotting Surface
% X1_new = zeros(res,res);
% Y1_new = zeros(res,res);

% for col = 1:res
%     for row = 1:res
%         X1_new(row,col) = min_values(1,varX) + (X1(row,col)*(max_values(1,varX)-min_values(1,varX)));
%         Y1_new(row,col) = min_values(1,varY) + (X1(row,col)*(max_values(1,varY)-min_values(1,varY)));
%     end
% end

% denormalization
 raw_data4training = input_data;
% create X vector
 for jj=1:res
    for ii=1:res % here the conversion from normalised to real data again is based on the fact you normalised between 0-1
        X1(jj,ii)= min(raw_data4training(:,varX)) + ii*((max(raw_data4training(:,varX)) - min(raw_data4training(:,varX)))/res);
    end
 end
% create Y vector
 for jj=1:res
    for ii=1:res
        Y1(ii,jj)= min(raw_data4training(:,varY))+  ii*((max(raw_data4training(:,varY)) - min(raw_data4training(:,varY)))/res);
    end
 end
 

% R=corrplot(rawdata);


% surf(X1,Y1,Z1,'FaceAlpha',0.5);

% zmax=max(max(Z1));
% [id_xmax,id_ymax]=find(Z1==zmax);
% xmax=X1(id_xmax);
% ymax=Y1(id_ymax);
% hold on;
% plot3(xmax,ymax,zmax,'k.','markersize',20);
% text(xmax,ymax,zmax,[' x=',num2str(xmax),char(10),' y=',num2str(ymax),char(10),' z=',num2str(zmax),char(10),]);


% scatter3(X1,Y1,Z1,'filled','ColorVariable');
% scatter(X1,Z1,'filled');

% xlabel('ES');
% ylabel('ES');
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