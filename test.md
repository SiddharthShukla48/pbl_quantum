## Variable Naming (exactly what we mean)

For each exam $i$ and slot $k$:

- $x_{ik} \in \{0,1\}$: exam-slot assignment variable  
- $x_{ik} = 1$ means exam $i$ is placed in slot $k$

For capacity term in each slot $k$:

- $e_i$: enrollment (students) of exam $i$  
- $C$: slot capacity (your case: $C = 5500$)  
- $s_{kb} \in \{0,1\}$: slack bit $b$ for slot $k$, with weight $2^b$  
- $S_k = \sum_b 2^b s_{kb}$: slack value in slot $k$  
- $L_k = \sum_i e_i x_{ik}$: total load in slot $k$

Penalty weights:

- $\lambda_1$: one-hot penalty (your case: $10000$)  
- $\lambda_4$: capacity penalty (your case: $200$)  

---

## Why C1 should enforce one-hot (in isolation)

C1:

$$
E_1 = \lambda_1 \sum_i \left(1 - \sum_k x_{ik} \right)^2
$$

For one exam $i$, let $m_i = \sum_k x_{ik}$. Then contribution is:

$$
E_{1,i} = \lambda_1 (1 - m_i)^2
$$

Minimum is at $m_i = 1$.  
So C1 alone strongly wants **exactly one active slot per exam**.

---

## C4 term you are using

$$
E_4 = \lambda_4 \sum_k (L_k + S_k - C)^2
$$

This expands (per slot $k$) to:

$$
\lambda_4 \left(
L_k^2 + S_k^2 + 2 L_k S_k - 2 C L_k - 2 C S_k + C^2
\right)
$$

For any $x_{ik}$, diagonal coefficient gets:

$$
\lambda_4 (e_i^2 - 2 C e_i)
$$

For any slack bit $s_{kb}$, diagonal coefficient gets:

$$
\lambda_4 (2^{2b} - 2C \cdot 2^b)
$$

---

## Where scale imbalance appears

With $C = 5500$, $\lambda_4 = 200$, and even small $e_i$:

For $e_i = 1$:

$$
\lambda_4 (e_i^2 - 2 C e_i)
= 200 (1 - 11000)
\approx -2.2 \times 10^6
$$

For $e_i = 10$:

$$
200 (100 - 110000)
\approx -2.198 \times 10^7
$$

Compare with C1 scale:

- C1 diagonal is around $-10^4$  
- C1 same-exam off-diagonal is $+2 \times 10^4$  

So raw C4 coefficients are often **2–3 orders of magnitude larger than C1 coefficients**.

Even though quadratic terms partly counterbalance, this huge dynamic range can make the annealer prefer patterns that are bad for one-hot (multiple $x_{ik} = 1$ for same exam), because those patterns can still look energetically attractive under dominant C4 structure.

---

## That is exactly what your logs show:

- "Exam 0 has 4/5/6/7 active slots"  
- "Only 522/554 exams assigned"  

persists despite increasing reads.

---

## Why increasing num_reads doesn’t fix this

`num_reads` improves search over the same objective.

If the objective itself is badly scaled (C4 dominating C1), then more reads just find better minima of that flawed landscape, not necessarily valid one-hot schedules.

So:

- Reads help stochastic quality  
- Reads do **not** solve formulation/scale imbalance  

---

## “Same conflicts with 30 slots” — suspicious?

No, not suspicious.

Graph conflicts (dataset edges) is a property of the fixed input graph $A$, so it stays same unless data/filter/mode changes.

What should improve is:

- `solution_conflict_violations`  
- one-hot validity (Exam $i$ has ... active slots should disappear)

So same graph edges is expected; invalid one-hot is the true red flag.

---
