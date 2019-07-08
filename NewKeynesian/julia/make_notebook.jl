using JSON
const include_re = r"^include\(\"(.+?)\"\)$"
const module_re = r"^module\s.+"

function strip_module_end(content)
    content = strip(content)
    if endswith(content, "end  # module")
        content = strip(content[1:end-length("end  # module")])
    end
    content
end

function process_cell(cell_body)
    if startswith(cell_body, "#=") && endswith(cell_body, "=#")
        return md_cell(strip(match(r"#=(.+)=#"s, cell_body)[1]))
    end
    if startswith(cell_body, "\"\"\"") && endswith(cell_body, "\"\"\"")
        return md_cell(strip(match(r"\"\"\"(.+)\"\"\""s, cell_body)[1]))
    end
    return code_cell(strip_module_end(cell_body))
end

myd(;kwargs...) = Dict(string(k) => v for (k, v) in kwargs)

function md_cell(content)
    source = [line * "\n" for line in split(content, "\n")]
    source[end] = rstrip(source[end])
    myd(
        cell_type="markdown",
        metadata=myd(outputExpanded=false),
        source=source
    )
end

function code_cell(content)
    source = [line * "\n" for line in split(content, "\n")]
    source[end] = rstrip(source[end])
    myd(
        outputs=Any[],
        cell_type="code",
        source=source,
        metadata=myd(outputExpanded=false),
        execution_count=nothing
    )
end

# from a string that contains txt, build a cell
function _build_cell(txt::String)
    out = Dict[]
    pos = 1
    while true
        try
            cell_body, pos = get_cell(txt, pos)
            cell = process_cell(cell_body)
            if isa(cell, Dict)
                push!(out, process_cell(cell_body))
            else
                break
            end catch e
            if isa(e, BoundsError)
                break
            else
                rethrow(e)
            end
        end
    end
    out
end

function build_cells(fn::AbstractString)
    @assert isfile(fn)
    out = Dict[]

    # read in the file as a vector of lines
    lines = readlines(fn)

    # track which lines are important (start a cell or include a file)
    important_lines = Tuple{Int,Symbol}[]
    for (i, l) in enumerate(lines)
        if occursin(module_re, l)
            push!(important_lines, (i, :skip))
        end
        if l == "#%% cell"
            push!(important_lines, (i, :cell))
        end
        if l == "#%% skip"
            push!(important_lines, (i, :skip))
        end
        if occursin(include_re, l)
            push!(important_lines, (i, :include))
        end
    end

    # add the end of the file
    push!(important_lines, (length(lines)+1, :end))

    for (i, (l, typ)) in enumerate(important_lines[1:end-1])
        lp, typ_p = important_lines[i+1]
        if typ == :cell
            txt = strip(join(lines[l+1:lp-1], '\n'))
            push!(out, process_cell(txt))
        elseif typ == :include
            append!(out, build_cells(match(include_re, lines[l])[1]))
        elseif typ == :skip
            continue
        else
            error("Handle this type of important line")
        end
    end

    return out

end


function create_notebook(fn::String)
    @assert isfile(fn)
    cells = build_cells(fn)
    if endswith(basename(fn), ".jl")
        return nb = myd(
            nbformat=4,
            nbformat_minor=1,
            metadata = myd(
                kernelspec = myd(
                    name = "julia-1.1",
                    display_name = "Julia 1.1",
                    language="julia"
                ),
                language_info = myd(
                    file_extension = ".jl",
                    mimetype = "application/julia",
                    name = "julia",
                    version = "1.1"
                )
            ),
            cells=build_cells(fn)
        )
    elseif endswith(basename(fn), ".py")
        return myd(
            nbformat=4,
            nbformat_minor=1,
            metadata = myd(
                kernelspec = myd(
                    name = "python3",
                    display_name = "Python 3",
                    language="python"
                ),
                language_info = myd(
                    file_extension = ".py",
                    mimetype = "text/x-python",
                    name = "python",
                    nbconvert_exporter="python",
                    pygments_lexer="ipython3",
                    codemirror_mode=myd(
                        name="ipython",
                        version=3
                    ),
                    version="3.6.0"
                )
            ),
            cells=build_cells(fn)
        )
    end
end

function save_nb(fn, dest)
    open(dest, "w") do f
        print(f, JSON.json(create_notebook(fn)))
    end
end

function main()
    # save julia file
    this_dir = dirname(@__FILE__)
    nb_dir = joinpath(dirname(this_dir), "notebooks")
    py_dir = joinpath(dirname(this_dir), "python")

    save_nb(joinpath(this_dir, "main.jl"), joinpath(nb_dir, "julia.ipynb"))
    save_nb(joinpath(py_dir, "main.py"), joinpath(nb_dir, "python.ipynb"))
end
