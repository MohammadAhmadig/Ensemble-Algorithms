 function Logit_Boost(train,train_label)
    % Number of Iteration
    Cycle=500;
    k=10;
    % Size of train data
    [N,J]= size(train);
    % weights
    Wi=ones(N,1)/N;
    F=zeros(N,1)/N;
    h=ones(N,1);
    a1=max(train);
    a2=min(train);
    H=zeros(N,k*J);
    % Weak learning
    for i=1:k
        H(:,(i-1)*J+1:i*J) = train > ones(N,1)*(a2 + (a1 - a2).* (i/k));
    end

    d = zeros(1,k*J);
    d1 = zeros(1,k*J);
    d2 = zeros(1,k*J);
    iter=0;
    Fold=1e9;
    while max(max(abs(F - Fold)))>1e-5 & iter < Cycle
        iter = iter + 1
        Fold=F;
        S=train_label;
        y=(S>0)-(S<=0);
        yy=(y+1)/2;
        a = exp(F);
        a_minus = exp(-F);
        b = a + a_minus;
        p = a./(b+1e-6);
        Zi=(yy-p)./(p.*(1-p)+1e-5);
        Wi=p.*(1-p);
        H = H - (ones(N,1) * mean(H));
        d1 = (Wi.*Zi)' * H;
        d2 = Wi' * (H.^2);
        d = (d1.^2 ./ (d2+1e-20));
        im = min(find(d==max(d)));
        F = F + 0.5 * ((d1(im) / d2(im)) * H (:,im));
        F=max(-15,min(15,F));
        output=max(F);
        output=mean(((F>0).*(1-yy))+((F<=0).*yy));
        output
        [x(iter)]=(output);
        [v(iter)]=(iter);
        plot(v,x);
        grid on;
        %drawnow
        disp('end loop')
    end
end