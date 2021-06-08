# Standard R methods are used to plot the results summarized across the replicates:

library(tidyverse)

ymin = function(x) last(x[(ntile(x, 20)==1)])
ymax = function(x) last(x[(ntile(x, 20)==19)])

plot_sims <- function(sims_df)
  sims_df %>%
  group_by(time, model) %>%
  summarise(ymin = ymin(state),
            ymax = ymax(state),
            state = mean(state), .groups = "drop") %>%
  ggplot(aes(time, state, ymin = ymin, ymax = ymax, fill=model)) +
  geom_ribbon(alpha= 0.3) + geom_line(aes(col = model))

plot_policy <- function(policy_df)
  policy_df %>% ggplot(aes(state, action,
                           group=interaction(rep, model),
                           col = model)) +
  geom_line(show.legend = FALSE) +
  coord_cartesian(xlim = c(0, 1.2))

plot_reward <- function(sims_df, gamma = 1)
  sims_df %>%
  group_by(rep, model) %>%
  mutate(cum_reward = cumsum(reward * gamma^time)) %>%
  group_by(time, model) %>%
  summarise(mean_reward = mean(cum_reward),
            sd = sd(cum_reward), .groups = "drop") %>%
  ggplot(aes(time, mean_reward)) +
  geom_ribbon(aes(ymin = mean_reward - 2*sd,
                  ymax = mean_reward + 2*sd, fill = model),
              alpha=0.25, show.legend = FALSE) +
  geom_line(aes(col = model), show.legend = FALSE) +
  ylab("reward")
