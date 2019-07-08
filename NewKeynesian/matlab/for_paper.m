info = cpuinfo()

inputs = cell(1, 15);
i = 1;

for deg = 1:5
    inputs{i} = struct('piestar', 1.0, 'zlb', 1, 'Degree', deg); i = i + 1;
    inputs{i} = struct('piestar', 1.0, 'zlb', 0, 'Degree', deg); i = i + 1;
    inputs{i} = struct('piestar', 1.0 + 0.0598/4, 'zlb', 0, 'Degree', deg); i = i + 1;
end


warning('off', 'all');

f_csv = fopen('output_20190705.csv', 'w');
fprintf(f_csv, 'pi_star,zlb,degree,solve_time,l_1,l_inf\n');
for i = 1:length(inputs)
    in = inputs{i};
    fprintf('Starting with zlb =%2d, piestar =%15d, degree=%2d\n', in.zlb, in.piestar, in.Degree);
    [t, l1, linf] = main_extracted(in);
    fprintf('Results:\n\t-time =%2.4f\n\t-l1 =%15d\n\t-l_inf =%15d\n\n\n\n', t, l1, linf);

    fprintf(f_csv, '%f,%d,%d,%f,%f,%f\n', in.piestar, in.zlb, in.Degree, t, l1, linf);
end

fclose(f_csv);
