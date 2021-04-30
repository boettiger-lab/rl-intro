
<!-- scratch draft based on presubmission letter, needs referencing and rephrasing at least.... -->

<!-- Environmental problems, ML -->
Advances in both available data and computing power have begun to open the door to a greater role for machine learning (ML) in addressing some of our planet’s most pressing environmental problems, such as the growing frequency and intensity of wildfire, over-exploited fisheries, declining biodiversity, and zoonotic pandemics.
But will ML approaches really help us tackle a changing planet?
Applications of ML in ecology have begun to illustrate the promise of both supervised learning (e.g. image recognition) and unsupervised learning (including clustering and prediction), but have so far largely overlooked the third and possibly most promising approach in the ML triad: Reinforcement Learning (RL).
Supervised and unsupervised learning approaches often require massive amounts of historical data to learn underlying patterns.
Even when such data is eventually available, these two methods may prove reliable for relatively static tasks such as species identification from audio or visual recordings, yet be a poor guide to a future climate we have never observed.

<!-- RL is the ML for decision -->
The task of selecting actions in an uncertain and changing environment to maximize some objective is the explicit focus of RL.
To date, the uncertain and changing environments considered by RL research have largely been drawn from examples in robotic movement and video games.
RL agents have recently achieved superhuman performance on classic Atari games [@DQN], and described the success of Deepmind’s AlphaZero against grandmasters in Go, Chess, and Shogi [@alphazero].
We believe that similar techniques could improve decision-making in managing ecological environments as well.
Already, complex ecological simulators – SORTIE in forest ecosystems [@sortie], EcoPath with EcoSim in marine systems [ecopath], GCM models in climate [@gcm] -- are the Atari games of our policy grandmasters. <!-- yeah no -->
Detailed, process-based models allow a window into the future scenarios of ecological management.
Even simple ecological models can provide important insight into future scenarios that no amount of historical data would reveal.
Rather than displacing existing ecological models and knowledge, we illustrate how RL-based approaches can allow us to leverage such understanding to better explore and guide complex decision-making in changing environments.
Whereas previous decision techniques have historically constrained the complexity and realism of practical modeling efforts, RL may open the door to the design of ever more realistic assumptions.
Drawing on examples from fisheries management, ecosystem tipping points, and wildfire spread, we illustrate how ecological models and simulations can be used to successfully train RL agents that match or exceed the performance of available management strategies.
Although our examples are primarily a proof-of-principle, they serve to illuminate both the promise as well as potential pitfalls.

Those pitfalls are not only technical, but include issues of ethics and power, particularly if the algorithms or data are proprietary.
We conclude with a discussion of how an open, transparent and reproducible approach can help mitigate some concerns, while also offering a more effective interface between teams of researchers from both ecological and computer sciences.
We include an extensive appendix with carefully annotated code which should allow readers to both reproduce and extend this analysis.
We further include implementations of three fully featured python modules following current leading standards, which would allow engineers and computer scientists already working in RL to test their agents against both solved and unsolved problems in conservation management.
We believe this piece would provide an effective introduction to the concepts and practices of RL which would allow ecologists to apply, extend, and critique such methods.




<!-- Scratch draft introduction -->


Deep Reinforcement Learning (RL), a machine learning technique in which a *software agent* is trained to select actions which best maximize some cumulative reward, could hold significant untapped promise for solving complex ecological and conservation decision-making problems from wildfire management to biodiversity preservation to zoonotic pandemics.
Unlike more widely recognized machine learning techniques for either *supervised* learning (such as image classification, [@species_ex], [@satellite_ex]) or unsupervised learning, RL focuses on solving *dynamic* problems which require planning ahead, often in uncertain and changing environments.
Of all machine learning techniques, RL most resembles a general artificial intelligence.
Notable successes using RL include AlphaZero, in which a software agent learned to beat the world's best human players in games of chess, Go, and Shogi [@alphazero], and major advances in robotics, including decision-making in autonomous vehicles [@robotics_ex].
These tasks all require the agent to not only anticipate future states of it's environment, but also respond to those changes to influence future outcomes.

<!-- Millie help on this para? -->

If conservation challenges are typically complex and dynamics, existing approaches are largely simple and static.
The establishment of protected areas (cite MPAs, 30x30, etc) focuses on policies fixed in space and time to address challenges that are fixed in neither.
The endangered species act ... Ecosystem services ....
Such simplicity is in no small part due to the challenge associated in determining a Some reference to [@TomDietrich] on potential for CS-based application.

In contrast to other ML methods, RL does not require massive data sets for training, relying instead on model simulations based on the best available ecological data and knowledge of underlying processes.
Despite the expansion in data collection from remote sensors, citizen science, and other efforts, ecological data remains sparse and unevenly sampled [CITE].
Moreover, no amount of data can train supervised or unsupervised ML approaches can predict a future which has no analog in the historical record.
Mechanistic, process-based models remain our most powerful tool to explore scenarios of dynamic ecosystems in changing climates.
Methods from *optimal control theory* have long been used in conservation and resource economics literature [@Clark1990] to determine optimal policies given such models [@Polasky2011].
However, control theory approaches are also tightly constrained by the "curse of dimensionality" [@Mangel1980; @Marescot2013], limiting their application to only simple, stylized models.
Deep reinforcement learning offers a way forward.
Deep RL agents are trained not with mountains of data, but by interacting with simulators.
Most recent advances in Deep RL have illustrated algorithms trained by playing arcade game simulators such as Atari BreakOut, or physics-based simulations of robotics movement [@TD3; @SAC; @PPO; @A2C].
Simulating from complex models is far simpler than solving them.
The richest ecosystem models, such as the global circulation models (GCMs) used to generate future climate scenarios [@something], or individual-based models used to predict forest growth [@something] or wildfire spread [@something], are massive simulations.
Given sufficient computational power, RL can be trained and evaluated across a broad range of possible realistic scenarios well beyond the reach of classical methods.
In this manner, RL would compliment, rather than replace, existing empirical and theoretical research which would continue to form the basis for those simulations.

If RL has great promise, the approach also imposes great risk.
Lacking the tractability of classical methods, an RL agent can appear to solve a wide class of scenarios with ease, but may fail unexpectedly under seemingly similar conditions [@henderson2017].
Despite recent progress, Deep RL is a young field and stability and robustness of its methods remain largely open problems
. RL approaches are no panacea, and improper design or testing can easily lead to negative outcomes
. Technical issues are not the only pitfall of AI applications to conservation problems
. Giving algorithms direct influence over conservation policy may also raise ethical and political issues, particularly if those algorithms (or components thereof) are treated as proprietary intellectual property


In this paper, we introduce the fundamentals of deep reinforcement learning and illustrate with simple examples how these approaches can be applied to important conservation decision-making problems, from over-harvesting in global fisheries to ecosystem tipping points to wildfire dynamics.
These examples span a range from problems for which an optimal analytic solution is already known, to problems not previously solved with classical methods, to problems not yet solved by RL.
We include a detailed appendix with complete examples in both python and R for training RL algorithms, as well as three stand-alone module "gyms" implementing the each of the conservation problems presented here using a standard benchmark [@gym], which we believe can facilitate future contributions both from researchers in RL searching for interesting and important open problems as well as ecologists who may seek to make those simulations more realistic and precise to specific application areas.
We hope this paper serves as a port of entry to researchers and students from either field seeking to engage in work on RL applications to conservation.
We also hope these examples serve as illustration of both the potential and perils of such applications.



# Sequential Decision Problems and Reinforcement Learning

<!-- meh, not sure we need this? -->

Reinforcement learning focuses primarily on sequential decision problems: tasks in which a decision-maker or "agent" must repeatedly decide what action to take in response to new information about the "state" of the system they seek to manage, in order to maximize some objective (referred to as the reward or utility).
Any conservation problem that considers the future possible scenarios is a sequential decision problem: even scenarios that face a one-off decision such as whether declare some region as a protected area, or list or de-list a species as endangered, are sequential decision problems if they must consider the timing of their actions.
In contrast, decisions that do not consider the future, such as deciding which of a collection of possible regions should be prioritized based only on the present: how many species currently inhabit a region, how much each region would cost to purchase today, are not sequential problems.
Central to any sequential decision problem is the need to consider possible future states, the need to forecast.
Because the agent's actions will influence that future, agents facing sequential decision problems must further be able to forecast how each possible action available to them will change that forecast.
Consequentially, our lack of knowledge about the underlying models and inability to predict future environments in our rapidly changing world pose a fundamental challenge to conservation decision-making.

This framework raises two conceptual challenges.
First, sequential decision-making problem is distinct from the problem of model estimation.
A sequential decision problem will typically treat the model as given, typically with some uncertainty around the process and possibly the measurmeents and model estimates as well.
Thus, methods for solving sequential decision-making should not be mistaken as alternatives to methods designed for model estimation, such as regression, Bayesian heirarchical modeling, generalized additive models (GAMs) etc.
Second, methods for describing and solving sequential decision-making problems are often given different terms by different communities to describe the same thing.
Sequential decision problems are common to behavioral ecology [@Mangel], conservation [@Marescot2013] natural resource economics [@Clark1990], and some engineering fields [] where they are known as "optimal control" problems, while computer science, neuroscience, and robotics typically refer to them as 'reinfocement learning'.





