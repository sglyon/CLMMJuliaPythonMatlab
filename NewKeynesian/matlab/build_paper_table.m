pistar_vals = [1.0; 1+0.0598/4];
for pistar_ind = 1:2
    pistar = pistar_vals(pistar_ind);
    for deg = 1:5
        NK_Main(pistar, deg, 0, 0);
        fprintf('The above is for:\n')
        fprintf('Polynomial Degree %0.2d\n', deg);
        fprintf('And pistar %5.5f\n', pistar);
        fprintf('\n\n\n');
    end
end
