%% try manifold optimization
% ignore the first constraint 
function x = test_Man_0714(Pi,M,Ns,epsilon_sig,v_div)


manifold = complexcirclefactory(M);
problem.M = manifold;

v_init = v_div;
% problem.costgrad = @(v) mycostgrad(Pi, v,Ns,epsilon_sig); %

problem.cost = @mycost;

problem.egrad = @mygrad;

% checkgradient(problem);
 
% Solve.
[x, xcost, info, options] = trustregions(problem,v_init);

%semilogy([info.iter], -real([info.cost]), 'o-','linewidth',2);
%xlabel('Number of iterations');
%ylabel('Objective function value');
%grid on


% function [obj, objg] = mycostgrad(Pi, v,Ns,epsilon_sig)
%     obj = 0; objg = 0;
%     for it = 1:Ns
%         ppi = Pi(:,it); 
%         PPi = ppi*ppi';
%         tP = v'*PPi*v;
%         ppt = real(1+ epsilon_sig(it)*tP);
%         obj = obj - log2(ppt);
%         objg = objg - (2/(log(2)*ppt))*(PPi*v);
%     end
%     
% end

    function [obj] = mycost(v)
        obj = 0; 
        for it = 1:Ns
            ppi = Pi(:,it); 
            PPi = ppi*ppi';
            tP = v'*PPi*v;
            ppt = real(1+ epsilon_sig(it)*tP);
            obj = obj - log2(ppt);
        end  
    end


    function [objg] = mygrad(v)
        objg = 0;
        for it = 1:Ns
            ppi = Pi(:,it); 
            PPi = ppi*ppi';
            tP = v'*PPi*v;
            ppt = real(1+ epsilon_sig(it)*tP);      
            objg = objg - (2/(log(2)*ppt))*(epsilon_sig(it)*PPi*v);
        end

    end

end