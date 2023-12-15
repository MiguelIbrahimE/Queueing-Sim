1 Correlation of processes
In the first part of the project we are going to look at the correlation structure of various processes. Autoregressive processes
are characterized by the fact that each time point consists of a linear combination of the process at previous points in time
plus iid noise. For example, the simplest autoregressive process ( also referred to as an autoregressive process of order 1)
𝑋𝑘+1 = 𝑎𝑋𝑘 + 𝐵𝑘.
where {𝐵𝑛} is a sequence of iid and zero mean ‘noise’ variables according to some distribution (for simulations, you can
use Normal random variables). For reasons of stability, we impose |𝑎| < 1.
A second type of process, we will consider is a Markov on-off process. We take a general discrete-time Markov chain
{𝑌𝑘} on the state space ℳ of size 𝑀: = |ℳ| < ∞ states and with transition matrix 𝑃. A function 𝑔: ℳ → {0, 1} marks
each state as belonging to OFF (‘0’) or ON (‘1’). The process {𝑋𝑘} where 𝑋𝑘 = 𝑔(𝑌𝑘) is a discrete-time Markovian on-off
process.
In particular, we want to look at a discrete-time homogeneous birth-death process with transitions:
𝑃𝑖𝑗 =
⎧
⎨
⎩
𝑝− if 𝑖 > 0 and 𝑗 = 𝑖 − 1
𝑝+ if 𝑖 < 𝑀 − 1 and 𝑗 = 𝑖 + 1
1 − 𝑝+ if 𝑖 = 𝑗 = 0
1 − 𝑝− if 𝑖 = 𝑗 = 𝑀 − 1
1 − 𝑝− − 𝑝+ if 0 < 𝑖 = 𝑗 < 𝑀 − 1
0 otherwise.

We assume that the indices start from zero and that 𝑔(𝑖) = 𝟭(𝑖 ≤ 𝑚), where 𝑚 ∈ {0, . . . , 𝑀 − 1} is a constant.
2 A model for an intermittent energy source.
Consider an ON-OFF process modelling the availability of an intermittent energy source like a solar panel or a wind
turbine. As a simplification, we assume that the source is either running at maximum capacity 𝐶 (in Watts) or no energy
is produced at all.
We have also a Poisson stream with rate 𝜆 of jobs {𝐽𝑘} that need a variable time to finish and a certain wattage: 𝐽𝑘 =
(𝜏𝑘, 𝑊𝑘), where {𝜏𝑘} and {𝑊𝑘} are iid random variables. At times the source is unavailable, we can either reschedule after
exponential time with parameter 𝛾 incurring a cost 𝑐, or we can buy external energy at a fixed rate 𝑝 per Wh.

1 Recapitulation of the model in assignment S-2, now with a battery
We focus on a Markovian model for the availability of an intermittent energy source like a solar panel or a wind turbine:
Assume there is a continuous-time Markov chain 𝑌(𝑡) with 𝑀 states and transition rate matrix 𝑄 of dimension 𝑀 × 𝑀.
A function 𝑔: {1, · · · , 𝑀} → {0, 1} maps the state of this Markov chain onto OFF (or ‘0’) and ON (or ‘1’). Then 𝑋(𝑡) =
𝑔(𝑌(𝑡)) is the ON-OFF process which models the intermittent energy source.
As before, we have a Poisson stream with rate 𝜆 of jobs {𝐽𝑘} that need a variable time to finish and a certain wattage: 𝐽𝑘 =
(𝜏𝑘, 𝑊𝑘), where {𝜏𝑘} and {𝑊𝑘} are iid random variables. At times when the source is unavailable, we can either reschedule
after exponential time with parameter 𝛾 incurring a cost 𝑐, or we can buy external energy at a fixed rate 𝑝 per Wh. Assume
that now we also have a battery at our disposal, of maximum capacity 𝐵 Wh to which you can store excess production at
an efficiency of 𝑒, 0 < 𝑒 < 1, that is, 1 Wh of offloaded energy results in 𝑒 Wh stored energy.
