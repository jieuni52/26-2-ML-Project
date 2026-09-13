# Lab 0 — Warm-up: Version Control and Six ML Equations

**Machine Learning Project (53744-01) · Week 1 · Due: Wednesday, 9 September 2026,**
**11:59 PM (KST), via e-Class**

> **This is a practice lab.** It is not graded out of 100 like Labs 1–12. It counts toward
> **participation credit on a pass/fail**. Nothing here is submitted as a zip —
> you submit a link to your own GitHub repository.
>

## 1. Goal

Two things, both of which every later lab takes for granted:

1. **Set up a GitHub account and a repository**, and put code in it with git.
2. **Implement six core ML equations from their definitions** — sigmoid, softmax,
   entropy, cross-entropy, KL divergence, and focal loss — with NumPy (and softmax once
   more in pure Python), then check them against the identities they must satisfy.

The last one, focal loss, you implement **by reading the paper it comes from**.

## 2. Environment


| Item             | Value                                                             |
| ------------------ | ------------------------------------------------------------------- |
| Python           | 3.10.x                                                            |
| Install          | `pip install -r requirements.txt` (numpy, pytest — nothing else) |
| Hardware         | CPU only                                                          |
| Expected runtime | < 5 seconds                                                       |
| Expected effort  | 2–3 hours (longer if you have never set up git authentication)   |
| Seed             | 42 (already set inside`main()`)                                   |

## 3. Task 1 — GitHub account and repository

1. Create a **GitHub account** if you do not have one: [https://github.com/signup](https://github.com/signup).
   Use an email address you will still be able to read after you graduate.
2. Create a **public repository**. The name is up to you — `machine-learning-project`
   is a sensible choice.
3. Put your finished Lab 0 work in it: a `lab00/` folder containing `src/` and `tests/`.
4. **Commit at least twice.** One commit for the initial files, at least one more for
   your implementation. A repository with a single "upload files" commit does not show
   that you used git; it shows that you used a web form.
5. Push to GitHub and check that the code is visible in the browser.

If you have never used git: GitHub's own quickstart is the shortest path —
[https://docs.github.com/en/get-started/quickstart](https://docs.github.com/en/get-started/quickstart). Work through "Create a repository"
and "Set up Git". Do not skip to a GUI without reading what `commit` and `push` mean;
you will need both vocabulary and commands in Lab 11-2.

**Do not commit** `__pycache__/`, `.pytest_cache/`, or a virtual environment. The
`.gitignore` shipped in this folder already excludes them — copy it into your repository.

## 4. Task 2 — Implement six equations

Open `src/lab00.py` and complete the seven TODO blocks. The full specification of each
function — arguments, return type, conventions, and the properties it must satisfy — is
in its docstring. Read them; they are the assignment.


| Task | Function                         | What it is                                                  |
| ------ | ---------------------------------- | ------------------------------------------------------------- |
| 1    | `sigmoid(z)`                     | logistic function, element-wise                             |
| 2    | `softmax_loop(z)`                | softmax in**pure Python** — lists, loops, `math`; no NumPy |
| 3    | `softmax_np(z)`                  | the same softmax, vectorised with NumPy, no Python loop     |
| 4    | `entropy(p)`                     | Shannon entropy H(p), in nats                               |
| 5    | `cross_entropy(p, q)`            | cross-entropy H(p, q), in nats                              |
| 6    | `kl_divergence(p, q)`            | Kullback–Leibler divergence D_KL(p‖q), in nats            |
| 7    | `focal_loss(p, q, gamma, alpha)` | focal loss —**from the paper**, see below                  |

Tasks 2 and 3 are the same equation twice. Write the loop version first, then the
vectorised one; a test checks that the two agree. `main()` times both, so you can see
what vectorisation is worth.

**Task 7 — read the paper.** Focal loss comes from
Lin et al., *Focal Loss for Dense Object Detection*, ICCV 2017:
[https://arxiv.org/abs/1708.02002](https://arxiv.org/abs/1708.02002). **Read Section 3.1 and Equations (4)–(5) only** —
about one page. The equation is not reproduced in this README or in the docstring —
take it from the paper. The docstring tells you how to map the paper's binary form onto
the K-class arguments you are given.

**Skeleton rules** — the tests depend on these:

- Write your code **only inside the TODO blocks**. You may add private helper functions.
- **Do not rename functions or change their arguments and return types.** The tests call
  them directly; a renamed function fails its tests.
- Do not modify `main()` or anything in `tests/`.
- Do not hard-code the values that appear in the tests. Implement the equations.

## 5. Run & self-check

```bash
pip install -r requirements.txt
python src/lab00.py            # prints every function's output plus a timing comparison
python -m pytest tests/ -q     # 26 tests
```

All 26 tests must pass. Several of them do not check a hard-coded number but a
*relationship* your functions must satisfy — for example that
`kl_divergence(p, q)` equals `cross_entropy(p, q) - entropy(p)`, and that
`focal_loss` with `gamma=0` reduces to `cross_entropy`. If those fail while the
individual functions look right, one of the two sides is wrong; the identity tells you
they cannot both be correct.

## 6. What to submit

**One line on e-Class: the URL of your public GitHub repository.** For example:

```
https://github.com/your-username/machine-learning-project
```

No zip file, no report, no `results.json` for this lab. Make sure the repository is
**public**, or the TA cannot open it.

Late submissions receive no credit, as for every other lab in this course.

## 7. How this is credited

Pass/fail, toward participation credit. You pass if **all three** hold:

1. All 26 public tests pass in your repository (`python -m pytest tests/ -q`).
2. The repository URL you submitted is public and contains your `lab00/` code.
3. The repository has **at least two commits**.

There is no partial credit and no 100-point rubric for this lab.

## 8. Academic honesty & AI use

AI tools are allowed for concept explanations, debugging, and API lookup — the same
policy as every other lab. Submitting AI-generated code as-is for the TODO blocks is
not. Labs are individual work.

You must be able to **explain every line you submit**. The instructor or TA may ask for
a short oral walkthrough; not being able to explain your own code is treated as evidence
of misconduct.

## 9. Questions

e-Class Q&A board (preferred, since answers benefit everyone — do not post solution code).
