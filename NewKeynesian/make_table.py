import pandas as pd
idx = pd.IndexSlice

jl = pd.read_csv("julia/output.csv")
jl["language"] = "Julia CPU (s)"

py = pd.read_csv("python/output.csv")
py["language"] = "Python CPU (s)"

ml = pd.read_csv("matlab/output.csv")
ml["language"] = "Matlab CPU (s)"

df = pd.concat([jl, py, ml])
replacements = {
    "degree": {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th"},
    }
df.replace(replacements, inplace=True)

ind_cols = ["pi_star", "zlb", "degree", "language"]
df.set_index(ind_cols, inplace=True)
df.sort_index(inplace=True)
df = df.unstack(level="language")


def part_to_tex(subset):
    times = subset["solve_time"].copy()
    for c in ["l_1", "l_inf"]:
        times[c] = subset.loc[:, (c, "Matlab CPU (s)")]
    col_order = ["l_1", "l_inf", "Julia CPU (s)", "Matlab CPU (s)", "Python CPU (s)"]
    times = times[col_order]
    times = times.reset_index(level=("pi_star", "zlb"), drop=True)
    txt = times.to_latex(float_format="%2.2f")
    return "\n".join(txt.splitlines()[5:10])


odd_pi_data = part_to_tex(df.loc[idx[1.01495, 0, :], :])
unit_pi_data = part_to_tex(df.loc[idx[1, 0, :], :])
unit_pi_zlb_data = part_to_tex(df.loc[idx[1, 1, :], :])


output = (r"""
    \begin{tabular}{c|cc|ccc}
    \hline \hline
    Degree & $L_1$ & $L_\infty$ & Julia CPU (s) & Matlab CPU (s) & Python CPU (s) \\ \hline
    \multicolumn{6}{c}{} \\
    \multicolumn{6}{c}{Inflation target $\pi_*=1.0598$} \\ \hline
    """ + odd_pi_data +
    r"""  \hline
    \multicolumn{6}{c}{} \\
    \multicolumn{6}{c}{Inflation target $\pi_*=1$} \\ \hline
    """ + unit_pi_data +
    r"""\\ \hline
    \multicolumn{6}{c}{} \\
    \multicolumn{6}{c}{Inflation target $\pi_*=1$ with ZLB} \\ \hline
    """ + unit_pi_zlb_data +
    """ \hline
    \end{tabular}
    """)

print("put this in the paper!!\n\n")
print(output)
