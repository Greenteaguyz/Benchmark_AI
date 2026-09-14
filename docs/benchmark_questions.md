# Fifteen-Question Reasoning Benchmark & Answer Key

**Author:** Hout Chanvireak (Benchmark & Evaluation Lead)  
**Collaborator:** Sok Ratanakvichea (Deployment & Integration Lead)  
**Date Frozen:** 2026-09-14  
**Status:** **FROZEN & APPROVED** (Conforming to Internship Project Plan, Pages 3 & 4)

---

## 1. Benchmark Overview & Distribution

The benchmark consists of exactly 15 standardized reasoning prompts divided equally into three categories (5 questions each) with calibrated difficulty tiers (2 Easy, 2 Medium, 1 Hard per category):

| Category | Question ID Range | Distribution | Primary Reasoning Skills Tested |
| :--- | :---: | :---: | :--- |
| **Mathematical Reasoning** | `Q01` – `Q05` | 2 Easy, 2 Med, 1 Hard | Linear algebra, percentage discounts, prime factorization / trailing zeroes, sequence rules, multi-dice combinatorics |
| **Logical Reasoning** | `Q06` – `Q10` | 2 Easy, 2 Med, 1 Hard | Semantic deduction, formal syllogisms, relative spatial ordering, classic river/water jug pouring, truth-teller/liar Knights & Knaves proof |
| **Programming Reasoning** | `Q11` – `Q15` | 2 Easy, 2 Med, 1 Hard | Clean function design, Python mutable reference semantics, binary search boundary bug detection, asymptotic recurrence, sliding window algorithm |

---

## 2. Fifteen Questions Bank, Verified Keys & Reasoning Points

### Category 1: Mathematical Reasoning

#### [Q01] Linear Equation Calculation
- **Difficulty:** Easy
- **Prompt:** `Solve for x: 3x + 12 = 30. Show each step of your calculation clearly and state the final numerical value of x.`
- **Verified Final Answer:** `x = 6`
- **Expected Reasoning Points:**
  1. Subtract 12 from both sides of the equation: `3x = 30 - 12 = 18`.
  2. Divide both sides by 3: `x = 18 / 3`.
  3. Clearly state the final numerical value `x = 6`.

#### [Q02] Percentage Discount
- **Difficulty:** Easy
- **Prompt:** `A store offers a 20% discount on an item originally priced at $80. What is the final sale price? Show your work.`
- **Verified Final Answer:** `$64`
- **Expected Reasoning Points:**
  1. Calculate the discount amount: `20% of $80 = 0.20 * 80 = $16` (or alternatively calculate `80% of $80 = 0.80 * 80`).
  2. Subtract discount from original price: `$80 - $16 = $64`.
  3. Conclude with final sale price of `$64`.

#### [Q03] Factorial Trailing Zeroes (Legendre's Formula)
- **Difficulty:** Medium
- **Prompt:** `Calculate the total number of trailing zeroes in 100! (100 factorial). Explain the mathematical reasoning behind your method.`
- **Verified Final Answer:** `24 trailing zeroes`
- **Expected Reasoning Points:**
  1. Trailing zeroes are produced by prime factors of 10 (`2 * 5`). Factors of 2 are abundant, so count factors of 5.
  2. Apply Legendre's formula: `floor(100/5) + floor(100/25) = 20 + 4 = 24`.
  3. Explain that powers of 5 greater than 100 (`5^3 = 125`) contribute 0, yielding exactly 24.

#### [Q04] Quadratic / Polynomial Sequence Detection
- **Difficulty:** Medium
- **Prompt:** `Find the next two numbers in the sequence: 2, 6, 12, 20, 30, ... Explain the underlying rule governing the pattern.`
- **Verified Final Answer:** `42 and 56`
- **Expected Reasoning Points:**
  1. Method A: Second differences are constant: first differences are `+4, +6, +8, +10`, next are `+12` (`30 + 12 = 42`) and `+14` (`42 + 14 = 56`).
  2. Method B: General formula `n*(n+1)` for `n = 1, 2, 3, 4, 5`: `6*7 = 42`, `7*8 = 56`.
  3. Explicitly state both numbers: 42 and 56.

#### [Q05] Probability with Three Dice
- **Difficulty:** Hard
- **Prompt:** `Three fair standard six-sided dice are rolled simultaneously. What is the exact probability that the sum of the numbers rolled is equal to 10? Express your answer as a simplified fraction and explain each step.`
- **Verified Final Answer:** `1/8` (or `27/216`)
- **Expected Reasoning Points:**
  1. Total possible outcomes = `6^3 = 216`.
  2. Enumerate or use generating functions / partitions of 10 into 3 integers from 1 to 6 (e.g. permutations of (6,3,1)=6, (6,2,2)=3, (5,4,1)=6, (5,3,2)=6, (4,4,2)=3, (4,3,3)=3 -> sum = 27).
  3. Form the probability fraction `27 / 216` and simplify to `1/8`.

---

### Category 2: Logical Reasoning

#### [Q06] Linguistic Constraint & Subtraction Deduction
- **Difficulty:** Easy
- **Prompt:** `A farmer has 17 sheep, and all but 9 die. How many live sheep does the farmer have left? State your answer and briefly explain the logic.`
- **Verified Final Answer:** `9 live sheep`
- **Expected Reasoning Points:**
  1. Clarify the semantic meaning of "all but 9 die": exactly 9 sheep survive while `17 - 9 = 8` die.
  2. Avoid the common arithmetic error of subtracting 9 from 17 (`17 - 9 = 8`).
  3. Conclude decisively that 9 live sheep remain.

#### [Q07] Syllogistic Quantifier Validity
- **Difficulty:** Easy
- **Prompt:** `All roses are flowers. Some flowers fade quickly. Does it strictly follow that some roses fade quickly? Answer with 'Yes' or 'No' and provide a formal logical justification.`
- **Verified Final Answer:** `No`
- **Expected Reasoning Points:**
  1. Direct answer: No.
  2. Justify via formal set logic or Venn diagrams: The subset "flowers that fade quickly" intersects the set of "flowers", but not necessarily the subset of "roses".
  3. Identify the fallacy: Fallacy of the undistributed middle.

#### [Q08] Linear Seating Arrangement Deduction
- **Difficulty:** Medium
- **Prompt:** `Alice, Bob, and Charlie are sitting in a row. Alice is not on the far right. Bob is sitting to the immediate right of Alice. Who is sitting on the far left? Deduce the exact left-to-right seating arrangement step by step.`
- **Verified Final Answer:** `Alice is on the far left (Arrangement: Alice, Bob, Charlie)`
- **Expected Reasoning Points:**
  1. Bob is immediately to the right of Alice, forming a contiguous block `[Alice, Bob]`.
  2. Alice cannot be on the far right (pos 3), so `[Alice, Bob]` must occupy positions `(1, 2)`.
  3. Position 3 must be occupied by Charlie, resulting in order: Alice, Bob, Charlie.
  4. Explicitly state that Alice is sitting on the far left.

#### [Q09] State-Space Water Jug Problem
- **Difficulty:** Medium
- **Prompt:** `You have a 3-liter jug and a 5-liter jug, and an unlimited supply of water. How can you measure out exactly 4 liters using only these two jugs without any measuring scales? Detail every pour step.`
- **Verified Final Answer:** `4 liters measured in the 5-liter jug`
- **Expected Reasoning Points:**
  1. Sequence of steps:
     - Fill the 5-liter jug completely (5, 0).
     - Pour from 5-liter jug into 3-liter jug until full (leaving 2 liters in the 5L jug) (2, 3).
     - Empty the 3-liter jug (2, 0).
     - Pour the 2 liters from the 5L jug into the 3L jug (0, 2).
     - Fill the 5-liter jug completely again (5, 2).
     - Pour from 5L jug into 3L jug until full (needs 1 liter), leaving exactly 4 liters in the 5L jug (4, 3).
  2. Document state tracking after each pour.

#### [Q10] Knights and Knaves Proof by Cases
- **Difficulty:** Hard
- **Prompt:** `On an island, inhabitants are either Knights (who always tell the truth) or Knaves (who always lie). You meet two inhabitants, A and B. A says: 'At least one of us is a Knave.' Determine precisely what A is and what B is. Provide a rigorous proof by cases.`
- **Verified Final Answer:** `A is a Knight, and B is a Knave`
- **Expected Reasoning Points:**
  1. Case 1: Assume A is a Knave. A's statement ("At least one of us is a Knave") would be true since A is indeed a Knave. But Knaves cannot tell the truth—contradiction!
  2. Conclude that A must be a Knight, so A's statement must be true.
  3. Since A's statement is true, at least one of {A, B} is a Knave. Since A is a Knight, B must be a Knave.
  4. Final deduction: A is a Knight, and B is a Knave.

---

### Category 3: Programming Reasoning

#### [Q11] Palindrome Algorithm Design
- **Difficulty:** Easy
- **Prompt:** `Write a clean Python function is_palindrome(s: str) -> bool that checks if a string is a palindrome, ignoring non-alphanumeric characters and case. Provide a brief trace of why it works.`
- **Verified Final Answer:** Function normalizing characters with `.isalnum()` and `.lower()`, returning `clean == clean[::-1]` (or two-pointer approach).
- **Expected Reasoning Points:**
  1. Filter non-alphanumeric characters and normalize case (e.g. `[c.lower() for c in s if c.isalnum()]`).
  2. Compare filtered sequence with its reverse or use two pointers from ends moving inward.
  3. Correctly handles spaces, punctuation, and mixed casing.

#### [Q12] Python Reference Semantics & Mutation Trace
- **Difficulty:** Easy
- **Prompt:** `Trace the execution of this Python code snippet and determine the exact printed output:`
  ```python
  x = [1, 2, 3]
  y = x
  y.append(4)
  print(len(x), x == y)
  ```
  `Explain why len(x) is affected.`
- **Verified Final Answer:** `4 True`
- **Expected Reasoning Points:**
  1. `y = x` assigns a reference to the same list object in memory, not a shallow or deep copy.
  2. `y.append(4)` mutates the underlying list in-place.
  3. `x` and `y` reference the identical object `[1, 2, 3, 4]`.
  4. Printed output is exactly `4 True`.

#### [Q13] Binary Search Off-By-One Bug Detection
- **Difficulty:** Medium
- **Prompt:** `Identify the bug in the following binary search implementation and explain how to fix it:`
  ```python
  def binary_search(arr, target):
      low = 0
      high = len(arr)
      while low < high:
          mid = (low + high) // 2
          if arr[mid] == target:
              return mid
          elif arr[mid] < target:
              low = mid
          else:
              high = mid - 1
      return -1
  ```
- **Verified Final Answer:** `low = mid` causes an infinite loop when `high - low == 1` and `arr[mid] < target`; also `high = len(arr) - 1` boundary conventions.
- **Expected Reasoning Points:**
  1. Pinpoint the infinite loop flaw: when `low = mid`, integer division `(low + high) // 2` truncates, causing `low` never to advance if `target` is greater than `arr[mid]`.
  2. State correct assignment: `low = mid + 1`.
  3. Note standard boundary pairing: either `while low <= high:` with `high = len(arr) - 1` and `high = mid - 1`, OR `while low < high:` with `high = mid`.

#### [Q14] Fibonacci Algorithmic Complexity Comparison
- **Difficulty:** Medium
- **Prompt:** `Explain the time and space complexity of computing the Nth Fibonacci number using naive recursion versus dynamic programming (memoization). Show the recurrence relations and dynamic state array.`
- **Verified Final Answer:** Naive recursion: `O(2^N)` (or `O(phi^N)`) time, `O(N)` stack space. Dynamic programming: `O(N)` time, `O(N)` (or `O(1)`) space.
- **Expected Reasoning Points:**
  1. Naive recursion recurrence `T(N) = T(N-1) + T(N-2) + O(1)` generates an exponential call tree of depth N, taking `O(2^N)` time and `O(N)` call stack memory.
  2. Memoized / iterative DP computes each subproblem `F(0)` through `F(N)` exactly once, reducing time to `O(N)`.
  3. Space complexity for 1D memo array is `O(N)`, which can be optimized to `O(1)` by storing only the previous two states.

#### [Q15] Longest Substring Without Repeating Characters
- **Difficulty:** Hard
- **Prompt:** `Design an algorithm to find the longest substring without repeating characters in a given string s. State the time and space complexities, and provide a clean Python implementation with an inline trace.`
- **Verified Final Answer:** Sliding window using hash map/set running in `O(N)` time and `O(min(N, M))` space (where M is character set size).
- **Expected Reasoning Points:**
  1. Use a two-pointer sliding window `[left, right]` with a dictionary recording the last seen index of each character.
  2. When a duplicate character is encountered within the current window (`char_map[c] >= left`), update `left = char_map[c] + 1`.
  3. Maintain and return `max_length = max(max_length, right - left + 1)`.
  4. Complexity: `O(N)` time (each character processed at most twice), `O(min(N, Sigma))` auxiliary space.

---

## 3. Standard Response Scoring Rubric (Page 3 Verbatim)

| Evaluation Criterion | 2 Points (Full Credit) | 1 Point (Partial Credit) | 0 Points (No Credit) |
| :--- | :--- | :--- | :--- |
| **1. Final Answer** | Completely correct numerical value, code output, or conclusion. | Partly correct; minor computational or sign error. | Completely incorrect or missing. |
| **2. Reasoning Quality** | Clear, logically rigorous, step-by-step reasoning without gaps. | Some unclear, skipped, or weakly justified steps. | Illogical, incoherent, or unsupported steps. |
| **3. Instruction Following** | Fully satisfies every prompt constraint (format, steps, requested type). | Partly follows constraints; omits secondary formatting detail. | Fails to follow key prompt instructions. |
| **4. Factual Support** | No hallucinations or invented mathematical/logical claims. | Minor unsupported claim not invalidating final result. | Major hallucinations or fabricated theorems/methods. |

---

## 4. Benchmark Pilot Test & Freeze Record

- **Piloted Questions:** `Q01` (Math Easy), `Q06` (Logic Easy), `Q11` (Code Easy).
- **Tested Models:** `phi4-mini-reasoning`, `deepseek-r1:7b`, `qwen3:8b`.
- **Review Findings:**
  - `Q01`: Clear and unambiguous across all models.
  - `Q06`: Successfully discriminated between models that parse linguistic semantics ("all but 9") versus models performing blind arithmetic.
  - `Q11`: Function specification and typing constraints adhered to cleanly.
- **Freeze Certification:**
  - Both **Hout Chanvireak** and **Sok Ratanakvichea** confirm question wording, answer keys, and rubric are locked.
  - **Status: BENCHMARK FROZEN FOR 45-RESPONSE COLLECTION.**
