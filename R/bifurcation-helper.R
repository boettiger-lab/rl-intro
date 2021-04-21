library(purrr)

states <- seq(0,1.5, length=200)
p <- list(r=0.7, M=1.2, q=3, b=0.15, a=0.19, alpha=0.001,init_state=0.8)
may <-  function(y, p){
  y * p$r * (1 - y / p$M)  - p$a * y ^ p$q / (y ^ p$q + p$b ^ p$q)
}

## approximate the zeros by minimizing absolute value of function
f <- function(x, p) abs(may(x,p))


# use manual visuals to determine bifurcation value of A and relevant intervals
# for optimizer search of each equilibrium point
#tibble(x = states,f = may(x,p), ddx = ddx(x,p)) %>%
#  ggplot(aes(x,f)) + geom_line() + geom_line(aes(x,ddx), col = 2) + ylim(-.2, .2) + xlim(0,1)
#P1 <- p
#P1$a <- 0.165
#tibble(x = states,f = may(x,P1), abs = f(x,P1)) %>%
#  ggplot(aes(x,f)) + geom_line() + geom_line(aes(x,abs), col = 2) + ylim(-.05, .07) + xlim(0,1)
#P2 <- p
#P2$a <- 0.214
#tibble(x = states,f = may(x,P2), abs = f(x,P2)) %>%
#  ggplot(aes(x,f)) + geom_line() + geom_line(aes(x,abs), col = 2) + ylim(-.1, .1)+ xlim(0,1)


## Compute bifurcation plot
## Note that search intervals are tuned to parameter settings
## Also note we must explicitly handle search outside of bifurcation vale differently
temp <- seq(.155, .225, length.out = 50)
bifur <- function(a){
  p$a <- a
  ## only valid in range when both stable points exist
  low <- optimize(f, interval = c(0,0.28), p = p)$minimum
  threshold <- optimize(f, interval = c(0.22,0.6), p = p)$minimum
  high <- optimize(f, interval = c(0.4,1), p = p)$minimum
  if(a > 0.165 & a < 0.214) {
    data.frame(parameter=a,
               equilibrium = c("stable", "unstable", "stable"),
               group = c("low", "threshold", "high"),
               state = c(low, threshold, high))
  } else if (a <= 0.165){
    data.frame(parameter=a,
               equilibrium = c("stable", "unstable", "stable"),
               group = c("low", "threshold", "high"),
               state = c(NA, NA, high))
  } else if (a >= 0.214){
    data.frame(parameter=a,
               equilibrium = c("stable", "unstable", "stable"),
               group = c("low", "threshold", "high"),
               state = c(low, NA, NA))
  }
}

df <- purrr::map_dfr(temp,bifur)

readr::write_csv(df, "figs/bifur.csv")
#df %>%
#  ggplot(aes(parameter, state, lty=equilibrium, group = group)) + geom_line()
