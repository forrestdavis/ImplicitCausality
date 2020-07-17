library(ggplot2)
library(lme4)
library(lmerTest)
library(cowplot)
library(texreg)

path = "/home/forrestdavis/Projects/ImplicitCausality/"

rt_surp <- read.csv(paste(path, "results/Reading_Time_surp.csv", sep=''))
story_data <-read.csv(paste(path, "results/Story_Completion_score.csv", sep=''))
pronoun_surp <- read.csv(paste(path, "results/IC_mismatch_surp.csv", sep=''))

tf_who_sim <- read.csv(paste(path, "results/tf_who_flat_SIM.csv", sep=''))
tf_were_sim <- read.csv(paste(path, "results/tf_were_flat_SIM.csv", sep=''))

gpt_who_sim <- read.csv(paste(path, "results/gpt_who_flat_SIM.csv", sep=''))
gpt_were_sim <- read.csv(paste(path, "results/gpt_were_flat_SIM.csv", sep=''))

lstm_who_sim <- read.csv(paste(path, "results/LSTM_who_flat_SIM.csv", sep =''))
lstm_were_sim <- read.csv(paste(path, "results/LSTM_were_flat_SIM.csv", sep = ''))

tf_pronoun_sim <- read.csv(paste(path, "results/tf_pronoun_flat_SIM.csv", sep=''))
lstm_pronoun_sim <- read.csv(paste(path, "results/LSTM_pronoun_flat_SIM.csv", sep=''))
gpt_pronoun_sim <- read.csv(paste(path,"results/gpt_pronoun_flat_SIM.csv", sep=''))

#add categorical IC variable
pronoun_surp$hasIC <- pronoun_surp$bias>0
pronoun_surp$hasIC <- as.numeric(pronoun_surp$hasIC)

lstm_pronoun_sim$hasIC <- lstm_pronoun_sim$bias > 0
lstm_pronoun_sim$hasIC <- as.numeric(lstm_pronoun_sim$hasIC)

tf_pronoun_sim$hasIC <- tf_pronoun_sim$bias > 0
tf_pronoun_sim$hasIC <- as.numeric(tf_pronoun_sim$hasIC)

gpt_pronoun_sim$hasIC <- gpt_pronoun_sim$bias > 0
gpt_pronoun_sim$hasIC <- as.numeric(gpt_pronoun_sim$hasIC)

#########################
# Completion Surp Stats
#########################
compl_lstm_model <- lmer(LSTM_avg_score ~ hasIC*isHIGH + (1|item), data=story_data)
summary(compl_lstm_model)
anova(compl_lstm_model)
texreg(compl_lstm_model)

compl_tf_model <- lmer(tf_score ~ hasIC*isHIGH + (1|item), data=story_data)
summary(compl_tf_model)
anova(compl_tf_model)
texreg(compl_tf_model)

compl_gpt_model <- lmer(gpt_score ~ hasIC*isHIGH + (1|item), data=story_data)
summary(compl_gpt_model)
anova(compl_gpt_model)
texreg(compl_gpt_model)


#########################
# Reading Time Surp Stats
#########################
rt_lstm_model <- lmer(LSTM_avg_surp ~ hasIC*isHIGH*num + (1|item), data=rt_surp)
summary(rt_lstm_model)
anova(rt_lstm_model)
texreg(rt_lstm_model)

rt_tf_model <- lmer(tf_surp ~ hasIC*isHIGH*num + (1|item), data=rt_surp)
summary(rt_tf_model)
anova(rt_tf_model)
texreg(rt_tf_model)

rt_gpt_model <- lmer(gpt_surp ~ hasIC*isHIGH*num + (1|item), data=rt_surp)
summary(rt_gpt_model)
anova(rt_gpt_model)
texreg(rt_gpt_model)

#########################
# Reading Time Surp Plots
#########################

rt_surp$hasIC <- factor(rt_surp$hasIC)
rt_surp$isHIGH <- factor(rt_surp$isHIGH)

# LSTM
lstm_rt <- ggplot(rt_surp, aes(x=hasIC, y=LSTM_avg_surp, fill=isHIGH)) +
  geom_boxplot(notch=TRUE, outlier.size = 0.1) + ylim(0, 11)+ theme(text = element_text(size=18)) + 
  labs(x ="Verb Type", y = "LSTM Surprisal (RC verb)") + 
  scale_x_discrete(breaks=c("0","1"),
                       labels=c("non-IC", "IC")) + theme(legend.title = element_text(size=20), legend.text = element_text(size=20))+
  scale_fill_manual(values = c("#999999", "white"), name= "Agreement Location", labels=c("Low", "High"))

legend <- get_legend(lstm_rt)

lstm_rt <- lstm_rt + theme(legend.position='none')

# GPT-2 XL
gpt_rt <- ggplot(rt_surp, aes(x=hasIC, y=gpt_surp, fill=isHIGH)) +
  geom_boxplot(notch=TRUE, outlier.size = 0.1) + ylim(0, 11)+ theme(text = element_text(size=18)) + 
  labs(x ="Verb Type", y = "GPT-2 XL Surprisal (RC verb)") + 
  scale_x_discrete(breaks=c("0","1"),
                   labels=c("non-IC", "IC")) + theme(legend.position='none') +
  scale_fill_manual(values = c("#999999", "white"), name= "Type", labels=c("Low", "High"))

# TF-XL
tf_rt <- ggplot(rt_surp, aes(x=hasIC, y=tf_surp, fill=isHIGH)) +
  geom_boxplot(notch=TRUE, outlier.size=0.1) + ylim(0, 11)+ theme(text = element_text(size=18)) + 
  labs(x ="Verb Type", y = "TF-XL Surprisal (RC verb)") + 
  scale_x_discrete(breaks=c("0","1"),
                   labels=c("non-IC", "IC")) + theme(legend.position = 'none') +
  scale_fill_manual(values = c("#999999", "white"), name= "Type", labels=c("Low", "High"))

# Humans
set.seed(23)
good = rnorm(40, mean=2, sd=1)
bad = rnorm(40, mean=4, sd=1)
LowIC <- data.frame("surp" = bad, 'hasIC' = 1, "isHigh" = 0)
LownonIC <- data.frame("surp" = good, 'hasIC' = 0, "isHigh" = 0)
HighIC <- data.frame("surp" = good, 'hasIC' = 1, "isHigh" = 1)
HighnonIC <- data.frame("surp" = bad, 'hasIC' = 0, "isHigh" = 1)

human <- rbind(LowIC, LownonIC, HighIC, HighnonIC)
human$isHigh <- factor(human$isHigh)
human$hasIC <- factor(human$hasIC)

human_plot <- ggplot(human, aes(x=hasIC, y=surp, fill=isHigh)) +
  geom_boxplot(notch=TRUE, outlier.size = 0.1, outlier.shape = NA) + ylim(0, 12)+ ylim(0, 11)+ theme(text = element_text(size=18)) + 
  labs(x ="Verb Type", y = "Predicted Human Effect") + 
  scale_x_discrete(breaks=c("0","1"),
                   labels=c("non-IC", "IC")) + 
  scale_fill_manual(values = c("#999999", "white"), name= "Type", labels=c("Low", "High"))

human_plot <- human_plot + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank())

human_plot <- human_plot  + theme(legend.text = element_text(size=30),
                                  legend.title = element_text(size=30),
                                  legend.position = c(0.02, 0.98), 
                                  legend.justification = c("left", "top"), 
                                  legend.box.just = "center", 
                                  legend.margin = margin(6, 6, 6, 6))
plot_row <- plot_grid(lstm_rt, tf_rt)

title <- ggdraw() + 
  draw_label(
    "Relative Clause Attachment with Implicit Causality RC Verb Surprisal",
    fontface = 'bold',
    x = 0,
    hjust = 0
  ) +
  theme(
    # add margin on the left of the drawing canvas,
    # so title is aligned with left edge of first plot
    plot.margin = margin(0, 0, 0, 7)
  )
plot_row <- plot_grid(
  title, plot_row,
  ncol = 1,
  # rel_heights values control vertical title margins
  rel_heights = c(0.1, 1.2)
)

#Put them together 
rt_plots <- ggdraw() + draw_plot(plot_row, x=0, y= 0.5, width=1, height=0.5) +
  draw_plot(gpt_rt, x=0, y= 0, width=0.5, height=0.5) + 
  draw_plot(human_plot, x=0.5, y= 0, width=0.5, height=0.5) + 
  draw_plot_label(label = c("c)", "a)", "b)", "d)"), size = 15,
                  x = c(0.05, 0.05, 0.55, 0.55), y = c(0.07, 0.57, 0.57, 0.07))


#########################
# Reading Time SIM Stats
#########################

rt_sim_lstm_model <- lmer(sim ~ hasIC*NP*layer + (1|item), data=lstm_who_sim)
summary(rt_sim_lstm_model)
anova(rt_sim_lstm_model)
texreg(rt_sim_lstm_model)

lstm_who_NP1_IC <- subset(lstm_who_sim, hasIC==1 & NP==1 & layer==1)
lstm_who_NP2_IC <- subset(lstm_who_sim, hasIC==1 & NP==2 & layer==1)

lstm_who_NP1_nonIC <- subset(lstm_who_sim, hasIC==0 & NP==1 & layer==1)
lstm_who_NP2_nonIC <- subset(lstm_who_sim, hasIC==0 & NP==2 & layer==1)

t.test(lstm_who_NP1_IC$sim, lstm_who_NP1_nonIC$sim)
t.test(lstm_who_NP2_IC$sim, lstm_who_NP2_nonIC$sim)

rt_sim_tf_model <- lmer(sim ~ hasIC*NP*layer + (1|item), data=tf_who_sim)
summary(rt_sim_tf_model)
anova(rt_sim_tf_model)
texreg(rt_sim_tf_model, digits=3)

rt_sim_gpt_model <- lmer(sim ~ hasIC*NP*layer + (1|item), data=gpt_who_sim)
summary(rt_sim_gpt_model)
anova(rt_sim_gpt_model)
texreg(rt_sim_gpt_model, digits=4)

#########################
# Reading Time SIM Plots
#########################

#LSTM who
lstm_who_sim$hasIC <- factor(lstm_who_sim$hasIC)
lstm_who_sim <- subset(lstm_who_sim, num=='pl')
lstm_who_sim$NP <- factor(lstm_who_sim$NP)

lstm_who <- ggplot(lstm_who_sim, aes(x=factor(layer), y=sim, fill=interaction(NP, hasIC))) +
  geom_boxplot(notch=TRUE, outlier.size = 0.1) + ylim(0, 1) + theme(text = element_text(size=12)) + 
  labs(x ="Hidden Layer", y = "LSTM Similarity") + scale_fill_manual(values = c("#9999CC", "gold3", "darkorchid3", "darkgoldenrod4"), name= "Noun+Verb", labels=c("High+NonIC", "Low+NonIC", "High+IC", "Low+IC"))

lstm_who <- lstm_who  + theme(legend.position = 'none')

#TF-XL who
tf_who_sim$hasIC <- factor(tf_who_sim$hasIC)
tf_who_sim <- subset(tf_who_sim, num=='pl')
tf_who_sim$NP <- factor(tf_who_sim$NP)

tf_who <- ggplot(tf_who_sim, aes(x=factor(layer), y=sim, fill=interaction(NP, hasIC))) +
  geom_boxplot(notch=TRUE, outlier.size = 0.1) + ylim(0, 1) + theme(text = element_text(size=12)) + theme(legend.position='none') + 
  labs(x ="Hidden Layer", y = "TF-XL Similarity")+ scale_fill_manual(values = c("#9999CC", "gold3", "darkorchid3", "darkgoldenrod4"), name= "Noun+Verb", labels=c("High+NonIC", "Low+NonIC", "High+IC", "Low+IC"))


#GPT-2 XL who
gpt_who_sim$hasIC <- factor(gpt_who_sim$hasIC)
gpt_who_sim <- subset(gpt_who_sim, num=='pl')
gpt_who_sim$NP <- factor(gpt_who_sim$NP)
gpt_who_sim_small <- subset(gpt_who_sim, layer %%3 == 0)

gpt_who <- ggplot(gpt_who_sim_small, aes(x=factor(layer), y=sim, fill=interaction(NP, hasIC))) +
  geom_boxplot(notch=TRUE, outlier.size=0.1) + ylim(0, 1) + theme(text = element_text(size=12)) + theme(legend.position='none') + 
  labs(x ="Hidden Layer", y = "GPT-2 XL Similarity") + scale_fill_manual(values = c("#9999CC", "gold3", "darkorchid3", "darkgoldenrod4"), name= "Noun+Verb", labels=c("High+NonIC", "Low+NonIC", "High+IC", "Low+IC"))


set.seed(42)
good = rnorm(40, mean=4, sd=1)
bad = rnorm(40, mean=2, sd=1)
HighnonIC <- data.frame("sim" = bad, "NP" = 1, 'hasIC' = 0)
LownonIC <- data.frame("sim" = good, "NP" = 2, 'hasIC' = 0)
HighIC <- data.frame("sim" = good, "NP" = 1, 'hasIC' = 1)
LowIC <- data.frame("sim" = bad, "NP" = 2, 'hasIC' = 1)

human <- rbind(HighnonIC, LownonIC, HighIC, LowIC)
human$hasIC <- factor(human$hasIC)
human$NP <- factor(human$NP)

human_plot <- ggplot(human, aes(y=sim, fill=interaction(NP, hasIC))) +
  geom_boxplot(notch=TRUE, outlier.size=0.1, outlier.shape=NA) + ylim(0, 12) + theme(text = element_text(size=12)) +
  labs(x ="Hidden Layer", y = "Predicted Human Effect") + 
  scale_fill_manual(values = c("#9999CC", "gold3", "darkorchid3", "darkgoldenrod4"), name= "Noun+Verb", labels=c("High+NonIC", "Low+NonIC", "High+IC", "Low+IC"))

human_plot <- human_plot + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), 
                                 axis.text.x=element_blank(), axis.ticks.x=element_blank())

human_plot <- human_plot  + theme(legend.position = c(0.02, 0.98), 
                                  legend.justification = c("left", "top"), 
                                  legend.box.just = "center", 
                                  legend.margin = margin(6, 6, 6, 6))


plot_row <- plot_grid(lstm_who, tf_who, rel_widths =c(1, 1.8))

title <- ggdraw() + 
  draw_label(
    "Relative Clause Attachment with Implicit Causality who Similarity",
    fontface = 'bold',
    x = 0,
    hjust = 0
  ) +
  theme(
    # add margin on the left of the drawing canvas,
    # so title is aligned with left edge of first plot
    plot.margin = margin(0, 0, 0, 7)
  )
plot_row <- plot_grid(
  title, plot_row,
  ncol = 1,
  # rel_heights values control vertical title margins
  rel_heights = c(0.1, 1.2)
)

rt_who_plots <- ggdraw() + draw_plot(plot_row, x=0, y= 0.5, width=1, height=0.5) +
  draw_plot(gpt_who, x=0, y= 0, width=0.8, height=0.5) + 
  draw_plot(human_plot, x=0.8, y=0, width=0.2, height=0.5) + 
  draw_plot_label(label = c("c)", "a)", "b)", "d)"), size = 15,
                  x = c(0.03, 0.03, 0.35, 0.79), y = c(0.07, 0.57, 0.57, 0.07))


#GPT-2 XL who big
gpt_who_sim$hasIC <- factor(gpt_who_sim$hasIC)
gpt_who_sim <- subset(gpt_who_sim, num=='pl')
gpt_who_sim$NP <- factor(gpt_who_sim$NP)

gpt_who <- ggplot(gpt_who_sim, aes(x=factor(layer), y=sim, fill=interaction(NP, hasIC))) +
  geom_boxplot(notch=TRUE, outlier.size=0.1) + ylim(0, 1) + theme(text = element_text(size=12)) + theme(legend.position='none') + 
  labs(x ="Hidden Layer", y = "GPT-2 XL Similarity") + scale_fill_manual(values = c("#9999CC", "gold3", "darkorchid3", "darkgoldenrod4"), name= "Noun+Verb", labels=c("High+NonIC", "Low+NonIC", "High+IC", "Low+IC"))

plot_row <- plot_grid(gpt_who, human_plot, rel_widths =c(2.5, 0.5))

title <- ggdraw() + 
  draw_label(
    "Relative Clause Attachment with Implicit Causality who Similarity",
    fontface = 'bold',
    x = 0,
    hjust = 0
  ) +
  theme(
    # add margin on the left of the drawing canvas,
    # so title is aligned with left edge of first plot
    plot.margin = margin(0, 0, 0, 7)
  )
plot_row <- plot_grid(
  title, plot_row,
  ncol = 1,
  # rel_heights values control vertical title margins
  rel_heights = c(0.1, 1.2)
)

###########
#RT were/was
############

#TF-XL were
tf_were_HIGH <- subset(tf_were_sim, isHIGH==1)
tf_were_LOW <- subset(tf_were_sim, isHIGH==0)

tf_were_h <- ggplot(tf_were_HIGH, aes(x=factor(layer), y=sim, fill=interaction(NP, hasIC))) +
  geom_boxplot(notch=TRUE, outlier.size = 0.1)  + ylim(-0.1, 1)+ theme(text = element_text(size=12))  + 
  labs(x ="Hidden Layer", y = "TF-XL Similarity with High Agree") + scale_fill_manual(values = c("#9999CC", "gold3", "darkorchid3", "darkgoldenrod4"), name= "Noun+Verb", labels=c("High+NonIC", "Low+NonIC", "High+IC", "Low+IC"))

tf_were_h <- tf_were_h  + theme(legend.position = c(0.05, 0.95), 
                              legend.justification = c("left", "top"), 
                              legend.box.just = "right", 
                              legend.margin = margin(6, 6, 6, 6))


tf_were_l <- ggplot(tf_were_LOW, aes(x=factor(layer), y=sim, fill=interaction(NP, hasIC))) +
  geom_boxplot(notch=TRUE, outlier.size = 0.1)  + ylim(-0.1, 1)+ theme(text = element_text(size=12))  + theme(legend.position = 'none') + 
  labs(x ="Hidden Layer", y = "TF-XL Similarity with Low Agree") + scale_fill_manual(values = c("#9999CC", "gold3", "darkorchid3", "darkgoldenrod4"), name= "Noun+Verb", labels=c("High+NonIC", "Low+NonIC", "High+IC", "Low+IC"))

set.seed(42)
best = rnorm(40, mean=6, sd=1)
good = rnorm(40, mean=4, sd=1)
ok = rnorm(40, mean=2.5, sd=1)
bad = rnorm(40, mean=1, sd=1)
  HighnonIC <- data.frame("sim" = good, "NP" = 1, 'hasIC' = 0)
  LownonIC <- data.frame("sim" = ok, "NP" = 2, 'hasIC' = 0)
  HighIC <- data.frame("sim" = best, "NP" = 1, 'hasIC' = 1)
  LowIC <- data.frame("sim" = bad, "NP" = 2, 'hasIC' = 1)
  
  human <- rbind(HighnonIC, LownonIC, HighIC, LowIC)
  human$hasIC <- factor(human$hasIC)
  human$NP <- factor(human$NP)
  
  human_H_plot <- ggplot(human, aes(y=sim, fill=interaction(NP, hasIC))) +
    geom_boxplot(notch=TRUE, outlier.size=0.1, outlier.shape=NA) + ylim(0, 8) + theme(text = element_text(size=12)) +
    labs(x ="Hidden Layer", y = "Predicted Human High Agree Effect") + 
    scale_fill_manual(values = c("#9999CC", "gold3", "darkorchid3", "darkgoldenrod4"), name= "Noun+Verb", labels=c("High+NonIC", "Low+NonIC", "High+IC", "Low+IC"))
  
  human_H_plot <- human_H_plot + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), 
                                   axis.text.x=element_blank(), axis.ticks.x=element_blank()) + theme(legend.position='none')
  
  set.seed(42)
  best = rnorm(40, mean=6, sd=1)
  good = rnorm(40, mean=4, sd=1)
  ok = rnorm(40, mean=2.5, sd=1)
  bad = rnorm(40, mean=1, sd=1)
  HighnonIC <- data.frame("sim" = bad, "NP" = 1, 'hasIC' = 0)
  LownonIC <- data.frame("sim" = best, "NP" = 2, 'hasIC' = 0)
  HighIC <- data.frame("sim" = ok, "NP" = 1, 'hasIC' = 1)
  LowIC <- data.frame("sim" = good, "NP" = 2, 'hasIC' = 1)
  
  human <- rbind(HighnonIC, LownonIC, HighIC, LowIC)
  human$hasIC <- factor(human$hasIC)
  human$NP <- factor(human$NP)
  
  human_L_plot <- ggplot(human, aes(y=sim, fill=interaction(NP, hasIC))) +
    geom_boxplot(notch=TRUE, outlier.size=0.1, outlier.shape=NA) + ylim(0, 8) + theme(text = element_text(size=12)) +
    labs(x ="Hidden Layer", y = "Predicted Human Low Agree Effect") + 
    scale_fill_manual(values = c("#9999CC", "gold3", "darkorchid3", "darkgoldenrod4"), name= "Noun+Verb", labels=c("High+NonIC", "Low+NonIC", "High+IC", "Low+IC"))
  
  human_L_plot <- human_L_plot + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), 
                                     axis.text.x=element_blank(), axis.ticks.x=element_blank()) + theme(legend.position = 'none')
  
  
  plot_row <- plot_grid(tf_were_h, human_H_plot, rel_widths =c(2.5, 0.8))
  plot_bottom_row <- plot_grid(tf_were_l, human_L_plot, rel_widths=c(2.5, 0.8))
  
  title <- ggdraw() + 
    draw_label(
      "Relative Clause Attachment with Implicit Causality RC Verb Similarity",
      fontface = 'bold',
      x = 0,
      hjust = 0
    ) +
    theme(
      # add margin on the left of the drawing canvas,
      # so title is aligned with left edge of first plot
      plot.margin = margin(0, 0, 0, 7)
    )
  plot_row <- plot_grid(
    title, plot_row,
    ncol = 1,
    # rel_heights values control vertical title margins
    rel_heights = c(0.1, 1.2)
  )
  
  tf_were_plots <- ggdraw() + draw_plot(plot_row, x=0, y= 0.5, width=1, height=0.5) +
    draw_plot(plot_bottom_row, x=0, y= 0, width=1, height=0.5) + 
    draw_plot_label(label = c("c)", "a)", "b)", "d)"), size = 15,
                    x = c(0.03, 0.03, 0.75, 0.75), y = c(0.07, 0.57, 0.57, 0.07))

 
#GPT-2 XL were
  gpt_were_HIGH <- subset(gpt_were_sim, isHIGH==1)
  gpt_were_LOW <- subset(gpt_were_sim, isHIGH==0)
  
  gpt_were_h <- ggplot(gpt_were_HIGH, aes(x=factor(layer), y=sim, fill=interaction(NP, hasIC))) +
    geom_boxplot(notch=TRUE, outlier.size = 0.1)  + ylim(-0.1, 1)+ theme(text = element_text(size=12))  + 
    labs(x ="Hidden Layer", y = "GPT-2 XL Similarity with High Agree") + scale_fill_manual(values = c("#9999CC", "gold3", "darkorchid3", "darkgoldenrod4"), name= "Noun+Verb", labels=c("High+NonIC", "Low+NonIC", "High+IC", "Low+IC"))
  
  gpt_were_h <- gpt_were_h  + theme(legend.position = c(0.05, 0.05), 
                                  legend.justification = c("left", "bottom"), 
                                  legend.box.just = "right", 
                                  legend.margin = margin(6, 6, 6, 6))
  
  
  gpt_were_l <- ggplot(gpt_were_LOW, aes(x=factor(layer), y=sim, fill=interaction(NP, hasIC))) +
    geom_boxplot(notch=TRUE, outlier.size = 0.1)  + ylim(-0.1, 1)+ theme(text = element_text(size=12))  + theme(legend.position = 'none') + 
    labs(x ="Hidden Layer", y = "GPT-2 XL Similarity with Low Agree") + scale_fill_manual(values = c("#9999CC", "gold3", "darkorchid3", "darkgoldenrod4"), name= "Noun+Verb", labels=c("High+NonIC", "Low+NonIC", "High+IC", "Low+IC"))
  
  set.seed(42)
  best = rnorm(40, mean=6, sd=1)
  good = rnorm(40, mean=4, sd=1)
  ok = rnorm(40, mean=2.5, sd=1)
  bad = rnorm(40, mean=1, sd=1)
  HighnonIC <- data.frame("sim" = good, "NP" = 1, 'hasIC' = 0)
  LownonIC <- data.frame("sim" = ok, "NP" = 2, 'hasIC' = 0)
  HighIC <- data.frame("sim" = best, "NP" = 1, 'hasIC' = 1)
  LowIC <- data.frame("sim" = bad, "NP" = 2, 'hasIC' = 1)
  
  human <- rbind(HighnonIC, LownonIC, HighIC, LowIC)
  human$hasIC <- factor(human$hasIC)
  human$NP <- factor(human$NP)
  
  human_H_plot <- ggplot(human, aes(y=sim, fill=interaction(NP, hasIC))) +
    geom_boxplot(notch=TRUE, outlier.size=0.1, outlier.shape=NA) + ylim(0, 8) + theme(text = element_text(size=12)) +
    labs(x ="Hidden Layer", y = "Predicted Human Effect") + 
    scale_fill_manual(values = c("#9999CC", "gold3", "darkorchid3", "darkgoldenrod4"), name= "Noun+Verb", labels=c("High+NonIC", "Low+NonIC", "High+IC", "Low+IC"))
  
  human_H_plot <- human_H_plot + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), 
                                       axis.text.x=element_blank(), axis.ticks.x=element_blank()) + theme(legend.position='none')
  
  set.seed(42)
  best = rnorm(40, mean=6, sd=1)
  good = rnorm(40, mean=4, sd=1)
  ok = rnorm(40, mean=2.5, sd=1)
  bad = rnorm(40, mean=1, sd=1)
  HighnonIC <- data.frame("sim" = bad, "NP" = 1, 'hasIC' = 0)
  LownonIC <- data.frame("sim" = best, "NP" = 2, 'hasIC' = 0)
  HighIC <- data.frame("sim" = ok, "NP" = 1, 'hasIC' = 1)
  LowIC <- data.frame("sim" = good, "NP" = 2, 'hasIC' = 1)
  
  human <- rbind(HighnonIC, LownonIC, HighIC, LowIC)
  human$hasIC <- factor(human$hasIC)
  human$NP <- factor(human$NP)
  
  human_L_plot <- ggplot(human, aes(y=sim, fill=interaction(NP, hasIC))) +
    geom_boxplot(notch=TRUE, outlier.size=0.1, outlier.shape=NA) + ylim(0, 8) + theme(text = element_text(size=12)) +
    labs(x ="Hidden Layer", y = "Predicted Human Effect") + 
    scale_fill_manual(values = c("#9999CC", "gold3", "darkorchid3", "darkgoldenrod4"), name= "Noun+Verb", labels=c("High+NonIC", "Low+NonIC", "High+IC", "Low+IC"))
  
  human_L_plot <- human_L_plot + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), 
                                       axis.text.x=element_blank(), axis.ticks.x=element_blank()) + theme(legend.position = 'none')
  
  
  plot_row <- plot_grid(gpt_were_h, human_H_plot, rel_widths =c(2.5, 0.5))
  plot_bottom_row <- plot_grid(gpt_were_l, human_L_plot, rel_widths=c(2.5, 0.5))
  
  title <- ggdraw() + 
    draw_label(
      "Relative Clause Attachment with Implicit Causality RC Verb Similarity",
      fontface = 'bold',
      x = 0,
      hjust = 0
    ) +
    theme(
      # add margin on the left of the drawing canvas,
      # so title is aligned with left edge of first plot
      plot.margin = margin(0, 0, 0, 7)
    )
  
  plot_row <- plot_grid(
    title, plot_row,
    ncol = 1,
    # rel_heights values control vertical title margins
    rel_heights = c(0.1, 1.2)
  )
  
  gpt_were_plots <- ggdraw() + draw_plot(plot_row, x=0, y= 0.5, width=1, height=0.5) +
    draw_plot(plot_bottom_row, x=0, y= 0, width=1, height=0.5) + 
    draw_plot_label(label = c("c)", "a)", "b)", "d)"), size = 15,
                    x = c(0.03, 0.03, 0.815, 0.815), y = c(0.07, 0.57, 0.57, 0.07))
  


#########################
# Reading Time were SIM Stats
#########################

rt_sim_lstm_model <- lmer(sim ~ hasIC*NP*layer*isHIGH + (1|item), data=lstm_were_sim)
summary(rt_sim_lstm_model)
anova(rt_sim_lstm_model)
texreg(rt_sim_lstm_model)

rt_sim_tf_model <- lmer(sim ~ hasIC*NP*layer*isHIGH + (1|item), data=tf_were_sim)
summary(rt_sim_tf_model)
anova(rt_sim_tf_model)
texreg(rt_sim_tf_model, digits=4)

rt_sim_gpt_model <- lmer(sim ~ hasIC*NP*layer*isHIGH + (1|item), data=gpt_were_sim)
summary(rt_sim_gpt_model)
anova(rt_sim_gpt_model)
texreg(rt_sim_gpt_model, digits=4)

#########################
# Pronoun Surp Stats
#########################

pronoun_surp$hasIC <- pronoun_surp$bias > 0
pronoun_surp$hasIC <- as.numeric(pronoun_surp$hasIC)

lstm_pronoun_model <- lmer(LSTM_avg_surp ~ hasIC*isHigh*gender + (1|item), data=pronoun_surp)
summary(lstm_pronoun_model)
anova(lstm_pronoun_model)
texreg(lstm_pronoun_model, digits=4)

tf_pronoun_model <- lmer(tf_surp ~ hasIC*isHigh*gender + (1|item), data=pronoun_surp)
summary(tf_pronoun_model)
anova(tf_pronoun_model)
texreg(tf_pronoun_model, digits=4)

gpt_pronoun_model <- lmer(gpt_surp ~ hasIC*isHigh*gender + (1|item), data=pronoun_surp)
summary(gpt_pronoun_model)
anova(gpt_pronoun_model)
texreg(gpt_pronoun_model, digits=4)

pronoun_HIGH <- subset(pronoun_surp, isHigh == 1)
pronoun_LOW <- subset(pronoun_surp, isHigh == 0)

pronoun_HIGH_m_IC <- subset(pronoun_HIGH, gender=='m' & hasIC==1)
pronoun_HIGH_m_nonIC <- subset(pronoun_HIGH, gender=='m' & hasIC==0)
pronoun_HIGH_f_IC <- subset(pronoun_HIGH, gender=='f' & hasIC==1)
pronoun_HIGH_f_nonIC <- subset(pronoun_HIGH, gender=='f' & hasIC==0)

t.test(pronoun_HIGH_m_IC$LSTM_avg_surp, pronoun_HIGH_m_nonIC$LSTM_avg_surp)
t.test(pronoun_HIGH_f_IC$LSTM_avg_surp, pronoun_HIGH_f_nonIC$LSTM_avg_surp)

t.test(pronoun_HIGH_m_IC$tf_surp, pronoun_HIGH_m_nonIC$tf_surp)
t.test(pronoun_HIGH_f_IC$tf_surp, pronoun_HIGH_f_nonIC$tf_surp)

t.test(pronoun_HIGH_m_IC$gpt_surp, pronoun_HIGH_m_nonIC$gpt_surp)
t.test(pronoun_HIGH_f_IC$gpt_surp, pronoun_HIGH_f_nonIC$gpt_surp)

pronoun_LOW_m_IC <- subset(pronoun_LOW, gender=='m' & hasIC==1)
pronoun_LOW_m_nonIC <- subset(pronoun_LOW, gender=='m' & hasIC==0)
pronoun_LOW_f_IC <- subset(pronoun_LOW, gender=='f' & hasIC==1)
pronoun_LOW_f_nonIC <- subset(pronoun_LOW, gender=='f' & hasIC==0)

t.test(pronoun_LOW_m_IC$LSTM_avg_surp, pronoun_LOW_m_nonIC$LSTM_avg_surp)
t.test(pronoun_LOW_f_IC$LSTM_avg_surp, pronoun_LOW_f_nonIC$LSTM_avg_surp)

t.test(pronoun_LOW_m_IC$tf_surp, pronoun_LOW_m_nonIC$tf_surp)
t.test(pronoun_LOW_f_IC$tf_surp, pronoun_LOW_f_nonIC$tf_surp)

t.test(pronoun_LOW_m_IC$gpt_surp, pronoun_LOW_m_nonIC$gpt_surp)
t.test(pronoun_LOW_f_IC$gpt_surp, pronoun_LOW_f_nonIC$gpt_surp)

#########################
# Pronoun Surp Plots
#########################

pronoun_surp$isHigh <- factor(pronoun_surp$isHigh)
pronoun_surp$gender <- factor(pronoun_surp$gender)
pronoun_surp$hasIC <- factor(pronoun_surp$hasIC)

# LSTM
lstm_pronoun <- ggplot(pronoun_surp, aes(x=hasIC, y=LSTM_avg_surp, fill=interaction(isHigh, gender))) +
  geom_boxplot(notch=TRUE, outlier.size = 0.1) + ylim(0, 6)+ theme(text = element_text(size=18)) + 
  labs(x ="Verb Bias", y = "LSTM Surprisal (Pronoun)") +
  scale_x_discrete(breaks=c("0","1"),
                   labels=c("Object-Bias", "Subject-Bias")) + 
  scale_fill_manual(values = c("#9999CC", "darkorchid3", "gold3", "darkgoldenrod4"), name= "Gender+Antecedent", labels=c("f+Obj", "f+Subj", "m+Obj", "m+Subj"))

#legend <- get_legend(lstm_pronoun) 
lstm_pronoun <- lstm_pronoun + theme(legend.position='none')

# GPT-2 XL
gpt_pronoun <- ggplot(pronoun_surp, aes(x=hasIC, y=gpt_surp, fill=interaction(isHigh, gender))) +
  geom_boxplot(notch=TRUE, outlier.size = 0.1) + ylim(0, 6)+ theme(text = element_text(size=18)) + 
  labs(x ="Verb Bias", y = "GPT-2 XL Surprisal (Pronoun)") +
  scale_x_discrete(breaks=c("0","1"),
                   labels=c("Object-Bias", "Subject-Bias")) + 
  scale_fill_manual(values = c("#9999CC", "darkorchid3", "gold3", "darkgoldenrod4"), name= "Gender+Antecedent", labels=c("f+Obj", "f+Subj", "m+Obj", "m+Subj"))

gpt_pronoun <- gpt_pronoun + theme(legend.position='none')

# TF-XL
tf_pronoun <- ggplot(pronoun_surp, aes(x=hasIC, y=tf_surp, fill=interaction(isHigh, gender))) +
    geom_boxplot(notch=TRUE, outlier.size = 0.1) + ylim(0, 6)+ theme(text = element_text(size=18)) + 
    labs(x ="Verb Bias", y = "TF-XL Surprisal (Pronoun)") +# theme(legend.position = 'none') +
  scale_x_discrete(breaks=c("0","1"),
                   labels=c("Object-Bias", "Subject-Bias")) + 
    scale_fill_manual(values = c("#9999CC", "darkorchid3", "gold3", "darkgoldenrod4"), name= "Gender+Antecedent", labels=c("f+Obj", "f+Subj", "m+Obj", "m+Subj"))
  
tf_pronoun <- tf_pronoun + theme(legend.position='none') 
  
# Humans
set.seed(23)
good = rnorm(40, mean=2, sd=1)
bad = rnorm(40, mean=4, sd=1)
fobjIC <- data.frame("surp" = bad, "gender" = 'f', 'hasIC' = 1, "isHigh" = 0)
fobjnonIC <- data.frame("surp" = good, "gender" = 'f', 'hasIC' = 0, "isHigh" = 0)
mobjIC <- data.frame("surp" = bad, "gender" = 'm', 'hasIC' = 1, "isHigh" = 0)
mobjnonIC <- data.frame("surp" = good, "gender" = 'm', 'hasIC' = 0, "isHigh" = 0)

fsubjIC <- data.frame("surp" = good, "gender" = 'f', 'hasIC' = 1, "isHigh" = 1)
fsubjnonIC <- data.frame("surp" = bad, "gender" = 'f', 'hasIC' = 0, "isHigh" = 1)
msubjIC <- data.frame("surp" = good, "gender" = 'm', 'hasIC' = 1, "isHigh" = 1)
msubjnonIC <- data.frame("surp" = bad, "gender" = 'm', 'hasIC' = 0, "isHigh" = 1)

human <- rbind(fobjIC, mobjIC, fobjnonIC,  mobjnonIC, fsubjIC, fsubjnonIC, msubjIC, msubjnonIC)
human$isHigh <- factor(human$isHigh)
human$hasIC <- factor(human$hasIC)
human$gender <- factor(human$gender)

human_plot <- ggplot(human, aes(x=hasIC, y=surp, fill=interaction(isHigh, gender))) +
  geom_boxplot(notch=TRUE, outlier.size = 0.1, outlier.shape = NA) + ylim(0, 12)+ theme(text = element_text(size=18)) + 
  labs(x ="Verb Bias", y = "Predicted Human Effect") +# theme(legend.position = 'none') +
  scale_x_discrete(breaks=c("0","1"),
                   labels=c("Object-Bias", "Subject-Bias")) + 
  scale_fill_manual(values = c("#9999CC", "darkorchid3", "gold3", "darkgoldenrod4"), name= "Gender+Antecedent", labels=c("f+Obj", "f+Subj", "m+Obj", "m+Subj"))

human_plot <- human_plot + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank())

human_plot <- human_plot  + theme(legend.position = c(0.02, 0.98), 
                                      legend.justification = c("left", "top"), 
                                      legend.box.just = "center", 
                                      legend.margin = margin(6, 6, 6, 6))

#plot together
plot_row <- plot_grid(lstm_pronoun, tf_pronoun)

title <- ggdraw() + 
  draw_label(
    "Pronoun Reference with Implicit Causality Surprisal",
    
    fontface = 'bold',
    x = 0,
    hjust = 0
  ) +
  theme(
    # add margin on the left of the drawing canvas,
    # so title is aligned with left edge of first plot
    plot.margin = margin(0, 0, 0, 7), text = element_text(size=25), 
  )
plot_row <- plot_grid(
  title, plot_row,
  ncol = 1,
  # rel_heights values control vertical title margins
  rel_heights = c(0.1, 1.2)
)

#Put them together 
pronoun_plots <- ggdraw() + draw_plot(plot_row, x=0, y= 0.5, width=1, height=0.5) +
  draw_plot(gpt_pronoun, x=0, y= 0, width=0.5, height=0.5) + 
  draw_plot(human_plot, x=0.5, y= 0, width=0.5, height=0.5) + 
  draw_plot_label(label = c("c)", "a)", "b)", "d)"), size = 15,
                  x = c(0.05, 0.05, 0.55, 0.55), y = c(0.07, 0.57, 0.57, 0.07))

#########################
# Pronoun SIM Stats
#########################
lstm_pronoun_sim$hasIC <- lstm_pronoun_sim$bias > 0
lstm_pronoun_sim$hasIC <- as.numeric(lstm_pronoun_sim$hasIC)

pronoun_sim_lstm_model <- lmer(sim ~ hasIC*NP*layer*gender + (1|item), data=lstm_pronoun_sim)
summary(pronoun_sim_lstm_model)
anova(pronoun_sim_lstm_model)
texreg(pronoun_sim_lstm_model, digits=4)

lstm_pronoun_HIGH_IC <- subset(lstm_pronoun_sim, hasIC==1 & layer == 2 & NP == 1)
lstm_pronoun_HIGH_nonIC <- subset(lstm_pronoun_sim, hasIC==0 & layer == 2 & NP == 1)

lstm_pronoun_LOW_IC <- subset(lstm_pronoun_sim, hasIC==1 & layer == 2 & NP == 2)
lstm_pronoun_LOW_nonIC <- subset(lstm_pronoun_sim, hasIC==0 & layer == 2 & NP == 2)

t.test(lstm_pronoun_HIGH_IC$sim, lstm_pronoun_HIGH_nonIC$sim)
t.test(lstm_pronoun_LOW_IC$sim, lstm_pronoun_LOW_nonIC$sim)


tf_pronoun_sim$hasIC <- tf_pronoun_sim$bias > 0
tf_pronoun_sim$hasIC <- as.numeric(tf_pronoun_sim$hasIC)

pronoun_sim_tf_model <- lmer(sim ~ hasIC*NP*layer*gender + (1|item), data=tf_pronoun_sim)
summary(pronoun_sim_tf_model)
anova(pronoun_sim_tf_model)
texreg(pronoun_sim_tf_model, digits=4)

tf_pronoun_HIGH_IC <- subset(tf_pronoun_sim, hasIC==1 & layer == 18 & NP == 1)
tf_pronoun_HIGH_nonIC <- subset(tf_pronoun_sim, hasIC==0 & layer == 18 & NP == 1)

tf_pronoun_LOW_IC <- subset(tf_pronoun_sim, hasIC==1 & layer == 18 & NP == 2)
tf_pronoun_LOW_nonIC <- subset(tf_pronoun_sim, hasIC==0 & layer == 18 & NP == 2)

t.test(tf_pronoun_HIGH_IC$sim, tf_pronoun_HIGH_nonIC$sim)
t.test(tf_pronoun_LOW_IC$sim, tf_pronoun_LOW_nonIC$sim)


gpt_pronoun_sim$hasIC <- gpt_pronoun_sim$bias > 0
gpt_pronoun_sim$hasIC <- as.numeric(gpt_pronoun_sim$hasIC)

pronoun_sim_gpt_model <- lmer(sim ~ bias*NP*layer*gender + (1|item), data=gpt_pronoun_sim)
summary(pronoun_sim_gpt_model)
anova(pronoun_sim_gpt_model)
texreg(pronoun_sim_gpt_model, digits=8)


gpt_pronoun_HIGH_IC <- subset(gpt_pronoun_sim, hasIC==1 & layer == 48 & NP == 1)
gpt_pronoun_HIGH_nonIC <- subset(gpt_pronoun_sim, hasIC==0 & layer == 48 & NP == 1)

gpt_pronoun_LOW_IC <- subset(gpt_pronoun_sim, hasIC==1 & layer == 48 & NP == 2)
gpt_pronoun_LOW_nonIC <- subset(gpt_pronoun_sim, hasIC==0 & layer == 48 & NP == 2)

t.test(gpt_pronoun_HIGH_IC$sim, gpt_pronoun_HIGH_nonIC$sim)
t.test(gpt_pronoun_LOW_IC$sim, gpt_pronoun_LOW_nonIC$sim)

#########################
# Pronoun SIM Plots
#########################

#LSTM pronoun
lstm_pronoun_sim$hasIC <- factor(lstm_pronoun_sim$hasIC)
lstm_pronoun_sim$NP <- factor(lstm_pronoun_sim$NP)

lstm_pronoun <- ggplot(lstm_pronoun_sim, aes(x=factor(layer), y=sim, fill=interaction(NP, hasIC))) +
  geom_boxplot(notch=TRUE, outlier.size=0.1) + ylim(0, 1) + theme(text = element_text(size=12)) + 
  labs(x ="Hidden Layer", y = "LSTM Similarity") + 
  scale_fill_manual(values = c("#9999CC", "gold3", "darkorchid3", "darkgoldenrod4"), name= "Antecedent+Verb Bias", labels=c("Subject+Obj Bias", "Object+Obj Bias", "Subject+Subj Bias", "Object+Subj Bias"))

lstm_pronoun <- lstm_pronoun  + theme(legend.position = 'none')

#TF-XL pronoun
tf_pronoun_sim$hasIC <- factor(tf_pronoun_sim$hasIC)
tf_pronoun_sim$NP <- factor(tf_pronoun_sim$NP)

tf_pronoun <- ggplot(tf_pronoun_sim, aes(x=factor(layer), y=sim, fill=interaction(NP, hasIC))) +
  geom_boxplot(notch=TRUE, outlier.size = 0.1) + ylim(0, 1) + theme(text = element_text(size=12)) + theme(legend.position='none') + 
  labs(x ="Hidden Layer", y = "TF-XL Similarity") + 
  scale_fill_manual(values = c("#9999CC", "gold3", "darkorchid3", "darkgoldenrod4"), name= "Noun+Verb Bias", labels=c("Subj+Obj Bias", "Obj+Obj Bias", "Subj+Subj Bias", "Obj+Subj Bias"))


#GPT-2 XL pronoun
gpt_pronoun_sim$hasIC <- factor(gpt_pronoun_sim$hasIC)
gpt_pronoun_sim$NP <- factor(gpt_pronoun_sim$NP)
gpt_pronoun_sim_small <- subset(gpt_pronoun_sim, layer %%3 == 0)

gpt_pronoun <- ggplot(gpt_pronoun_sim_small, aes(x=factor(layer), y=sim, fill=interaction(NP, hasIC))) +
  geom_boxplot(notch=TRUE, outlier.size=0.1) + ylim(0, 1) + theme(text = element_text(size=12)) + theme(legend.position='none') + 
  labs(x ="Hidden Layer", y = "GPT-2 XL Similarity") + 
  scale_fill_manual(values = c("#9999CC", "gold3", "darkorchid3", "darkgoldenrod4"), name= "Noun+Bias", labels=c("Subject+Obj Bias", "Object+Obj Bias", "Subject+Subj Bias", "Object+Subj Bias"))

# Humans
set.seed(42)
good = rnorm(40, mean=4, sd=1)
bad = rnorm(40, mean=2, sd=1)
SubjObj <- data.frame("sim" = bad, "NP" = 1, 'hasIC' = 0)
ObjObj <- data.frame("sim" = good, "NP" = 2, 'hasIC' = 0)
SubjSubj <- data.frame("sim" = good, "NP" = 1, 'hasIC' = 1)
ObjSubj <- data.frame("sim" = bad, "NP" = 2, 'hasIC' = 1)

human <- rbind(SubjObj, ObjObj, SubjSubj, ObjSubj)
human$hasIC <- factor(human$hasIC)
human$NP <- factor(human$NP)

human_plot <- ggplot(human, aes(y=sim, fill=interaction(NP, hasIC))) +
  geom_boxplot(notch=TRUE, outlier.size=0.1, outlier.shape=NA) + ylim(0, 12) + theme(text = element_text(size=12)) +
  labs(x ="Hidden Layer", y = "Predicted Human Effect") + 
  scale_fill_manual(values = c("#9999CC", "gold3", "darkorchid3", "darkgoldenrod4"), name= "Antecedent+Verb Bias", labels=c("Subject+Obj Bias", "Object+Obj Bias", "Subject+Subj Bias", "Object+Subj Bias"))

human_plot <- human_plot + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank(), 
                                 axis.text.x=element_blank(), axis.ticks.x=element_blank())

human_plot <- human_plot  + theme(legend.position = c(0.02, 0.98), 
                                  legend.justification = c("left", "top"), 
                                  legend.box.just = "center", 
                                  legend.margin = margin(6, 6, 6, 6))

plot_row <- plot_grid(lstm_pronoun, tf_pronoun, rel_widths =c(1, 1.8))

title <- ggdraw() + 
  draw_label(
    "Pronoun Reference with Implicit Causality Similarity",
    fontface = 'bold',
    x = 0,
    hjust = 0
  ) +
  theme(
    # add margin on the left of the drawing canvas,
    # so title is aligned with left edge of first plot
    plot.margin = margin(0, 0, 0, 7)
  )
plot_row <- plot_grid(
  title, plot_row,
  ncol = 1,
  # rel_heights values control vertical title margins
  rel_heights = c(0.1, 1.2)
)

pronoun_plots <- ggdraw() + draw_plot(plot_row, x=0, y= 0.5, width=1, height=0.5) +
  draw_plot(gpt_pronoun, x=0, y= 0, width=0.8, height=0.5) +
  draw_plot(human_plot, x=0.8, y=0, width=0.2, height=0.5) + 
  draw_plot_label(label = c("c)", "a)", "b)", "d)"), size = 15,
                  x = c(0.03, 0.03, 0.35, 0.79), y = c(0.07, 0.57, 0.57, 0.07))

#gpt-big
gpt_pronoun <- ggplot(gpt_pronoun_sim, aes(x=factor(layer), y=sim, fill=interaction(NP, hasIC))) +
  geom_boxplot(notch=TRUE, outlier.size=0.1) + ylim(0, 1) + theme(text = element_text(size=12)) + theme(legend.position='none') + 
  labs(x ="Hidden Layer", y = "GPT-2 XL Similarity") + 
  scale_fill_manual(values = c("#9999CC", "gold3", "darkorchid3", "darkgoldenrod4"), name= "Noun+Bias", labels=c("Subject+Obj Bias", "Object+Obj Bias", "Subject+Subj Bias", "Object+Subj Bias"))

plot_row <- plot_grid(gpt_pronoun, human_plot, rel_widths =c(2.5, 0.5))

title <- ggdraw() + 
  draw_label(
    "Pronoun Reference with Implicit Causality Similarity",
    fontface = 'bold',
    x = 0,
    hjust = 0
  ) +
  theme(
    # add margin on the left of the drawing canvas,
    # so title is aligned with left edge of first plot
    plot.margin = margin(0, 0, 0, 7)
  )
plot_row <- plot_grid(
  title, plot_row,
  ncol = 1,
  # rel_heights values control vertical title margins
  rel_heights = c(0.1, 1.2)
)
