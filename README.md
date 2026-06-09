# Collection of example OCPs to test changes made to FATROP

This repo collects some example problems that can be used to test modifications made to the FATROP solver.

## Generalized dynamics
The Generalized dynamics examples tackle the challenge of applying dynamics of the form $f(x_k, u_k, x_{k+1}) = 0$ instead of $x_{k+1} = f(x_k, u_k)$.
Such dynamics can be enforced by introducing auxiliary variables $z_k$ and auxiliary constraints $f(x_k, u_k, z_k) = 0$ leading to simple dynamics $x_{k+1} = z_k$.

However, any stage-wise equality constraints present before the reformulation do not depend on $z_k$ meaning there is a block-zero matrix in the stage-wise equality constraint Jacobian.
By passing the option <code>'problem_type':'accelerated_ocp_type'</code> (instead of the default value <code>'ocp_type'</code>), the LU decomposition of the stage-wise equality constraint Jacobian will exploit this structure.
For large equality constraint Jacobians, this can lead to a reduction of the computation time of the search direction of up to 20%.

The file <code>example_descriptions.md</code> explains some examples used to demonstrate this. The <code>hello world</code> example is a simple academic example that focusses on showing the syntax and usage easily. The other examples are more elaborated.
