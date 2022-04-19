function [Learners, Weights, final_hyp] = RealAdaBoost(WeakLrn, Data, Labels, Max_Iter, OldW, OldLrn, final_hyp)
Learners = OldLrn;
Weights = OldW;
final_hyp = Classify(Learners, Weights, Data);
distr = exp(- (Labels .* final_hyp));
distr = distr / sum(distr);

for It = 1 : Max_Iter
  nodes = train(WeakLrn, Data, Labels, distr);

  for i = 1:length(nodes)
    curr_tr = nodes{i};
    
    step_out = calc_output(curr_tr, Data); 
      
    s1 = sum( (Labels ==  1) .* (step_out) .* distr);
    s2 = sum( (Labels == -1) .* (step_out) .* distr);

    if(s1 == 0 && s2 == 0)
        continue;
    end
    Alpha = 0.5*log((s1 + eps) / (s2+eps));

    Weights(end+1) = Alpha;
    
    Learners{end+1} = curr_tr;
    
    final_hyp = final_hyp + step_out .* Alpha;    
  end
  
  distr = exp(- 1 * (Labels .* final_hyp));
  Z = sum(distr);
  distr = distr / Z;  

end
