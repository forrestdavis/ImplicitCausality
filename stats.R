library(ggplot2)
library(lme4)
library(lmerTest)
library(cowplot)

rt_surp <- read.csv("/Users/forrestdavis/projects/ImplicitCausality/results/Reading_Time_surp.csv")
story_data <-read.csv("/Users/forrestdavis/projects/ImplicitCausality/results/Story_Completion_score.csv")
pronoun_surp <- read.csv("/Users/forrestdavis/Projects/ImplicitCausality/results/IC_mismatch_surp.csv")

tf_who_sim <- read.csv("/Users/forrestdavis/projects/ImplicitCausality/results/tf_who_flat_SIM.csv")
tf_were_sim <- read.csv("/Users/forrestdavis/projects/ImplicitCausality/results/tf_were_flat_SIM.csv")

gpt_who_sim <- read.csv("/Users/forrestdavis/projects/ImplicitCausality/results/gpt_who_flat_SIM.csv")
gpt_were_sim <- read.csv("/Users/forrestdavis/projects/ImplicitCausality/results/gpt_were_flat_SIM.csv")

lstm_who_sim <- read.csv("/Users/forrestdavis/projects/ImplicitCausality/results/LSTM_who_flat_SIM.csv")
lstm_were_sim <- read.csv("/Users/forrestdavis/projects/ImplicitCausality/results/LSTM_were_flat_SIM.csv")

tf_pronoun_sim <- read.csv("/Users/forrestdavis/Projects/ImplicitCausality/results/tf_pronoun_flat_SIM.csv")
lstm_pronoun_sim <- read.csv("/Users/forrestdavis/Projects/ImplicitCausality/results/LSTM_pronoun_flat_SIM.csv")
gpt_pronoun_sim <- read.csv("/Users/forrestdavis/Projects/ImplicitCausality/results/gpt_pronoun_flat_SIM.csv")

#add categorical IC variable
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

compl_tf_model <- lmer(tf_score ~ hasIC*isHIGH + (1|item), data=story_data)
summary(compl_tf_model)
anova(compl_tf_model)

compl_gpt_model <- lmer(gpt_score ~ hasIC*isHIGH + (1|item), data=story_data)
summary(compl_gpt_model)
anova(compl_gpt_model)


#########################
# Reading Time Surp Stats
#########################
rt_lstm_model <- lmer(LSTM_avg_surp ~ hasIC*isHIGH*num + (1|item), data=rt_surp)
summary(rt_lstm_model)
anova(rt_lstm_model)

rt_tf_model <- lmer(tf_surp ~ hasIC*isHIGH*num + (1|item), data=rt_surp)
summary(rt_tf_model)
anova(rt_tf_model)

rt_gpt_model <- lmer(gpt_surp ~ hasIC*isHIGH*num + (1|item), data=rt_surp)
summary(rt_gpt_model)
anova(rt_gpt_model)

#########################
# Reading Time Surp Plots
#########################

# LSTM
lstm_rt <- ggplot(rt_surp, aes(x=factor(hasIC), y=LSTM_avg_surp, fill=factor(isHIGH))) +
  geom_boxplot(notch=TRUE) + ylim(0, 11)+ theme(text = element_text(size=12)) + 
  labs(x ="Verb Type", y = "LSTM Surprisal (RC verb)") + 
  scale_x_discrete(breaks=c("0","1"),
                       labels=c("non-IC", "IC")) + 
  scale_fill_manual(values = c("#999999", "white"), name= "Attachment Type", labels=c("Low", "High"))

legend <- get_legend(lstm_rt)

lstm_rt <- lstm_rt + theme(legend.position='none')

# GPT-2 XL
gpt_rt <- ggplot(rt_surp, aes(x=factor(hasIC), y=gpt_surp, fill=factor(isHIGH))) +
  geom_boxplot(notch=TRUE) + ylim(0, 11)+ theme(text = element_text(size=12)) + 
  labs(x ="Verb Type", y = "GPT-2 XL Surprisal (RC verb)") + 
  scale_x_discrete(breaks=c("0","1"),
                   labels=c("non-IC", "IC")) + theme(legend.position='none') +
  scale_fill_manual(values = c("#999999", "white"), name= "Type", labels=c("Low", "High"))

# TF-XL
tf_rt <- ggplot(rt_surp, aes(x=factor(hasIC), y=tf_surp, fill=factor(isHIGH))) +
  geom_boxplot(notch=TRUE) + ylim(0, 11)+ theme(text = element_text(size=12)) + 
  labs(x ="Verb Type", y = "TF-XL Surprisal (RC verb)") + 
  scale_x_discrete(breaks=c("0","1"),
                   labels=c("non-IC", "IC")) + theme(legend.position = 'none') +
  scale_fill_manual(values = c("#999999", "white"), name= "Type", labels=c("Low", "High"))

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
  draw_plot(legend, x=0.5, y= 0, width=0.5, height=0.5) + 
  draw_plot_label(label = c("c)", "a)", "b)"), size = 15,
                  x = c(0.05, 0.05, 0.55), y = c(0.07, 0.57, 0.57))

#########################
# Reading Time SIM Stats
#########################

rt_sim_lstm_model <- lmer(sim ~ hasIC*NP*layer + (1|item), data=lstm_who_sim)
summary(rt_sim_lstm_model)
anova(rt_sim_lstm_model)

lstm_who_NP1_IC <- subset(lstm_who_sim, hasIC==1 & NP==1 & layer==1)
lstm_who_NP2_IC <- subset(lstm_who_sim, hasIC==1 & NP==2 & layer==1)

lstm_who_NP1_nonIC <- subset(lstm_who_sim, hasIC==0 & NP==1 & layer==1)
lstm_who_NP2_nonIC <- subset(lstm_who_sim, hasIC==0 & NP==2 & layer==1)

t.test(lstm_who_NP1_IC$sim, lstm_who_NP1_nonIC$sim)
t.test(lstm_who_NP2_IC$sim, lstm_who_NP2_nonIC$sim)

rt_sim_tf_model <- lmer(sim ~ hasIC*NP*layer + (1|item), data=tf_who_sim)
summary(rt_sim_tf_model)
anova(rt_sim_tf_model)

rt_sim_gpt_model <- lmer(sim ~ hasIC*NP*layer + (1|item), data=gpt_who_sim)
summary(rt_sim_gpt_model)
anova(rt_sim_gpt_model)

#########################
# Reading Time SIM Plots
#########################

#LSTM who
lstm_who_sim$hasIC <- factor(lstm_who_sim$hasIC)
lstm_who_sim <- subset(lstm_who_sim, num=='pl')
lstm_who_sim$NP <- factor(lstm_who_sim$NP)

lstm_who <- ggplot(lstm_who_sim, aes(x=factor(layer), y=sim, fill=interaction(NP, hasIC))) +
  geom_boxplot(notch=TRUE) + ylim(0, 1) + theme(text = element_text(size=12)) + 
  labs(x ="Hidden Layer", y = "LSTM Similarity") + scale_fill_manual(values = c("#9999CC", "gold3", "darkorchid3", "darkgoldenrod4"), name= "Noun+Verb", labels=c("High+NonIC", "Low+NonIC", "High+IC", "Low+IC"))

lstm_who <- lstm_who  + theme(legend.position = c(0.05, 0.95), 
                            legend.justification = c("left", "top"), 
                            legend.box.just = "right", 
                            legend.margin = margin(6, 6, 6, 6))

#TF-XL who
tf_who_sim$hasIC <- factor(tf_who_sim$hasIC)
tf_who_sim <- subset(tf_who_sim, num=='pl')
tf_who_sim$NP <- factor(tf_who_sim$NP)

tf_who <- ggplot(tf_who_sim, aes(x=factor(layer), y=sim, fill=interaction(NP, hasIC))) +
  geom_boxplot(notch=TRUE) + ylim(0, 1) + theme(text = element_text(size=12)) + theme(legend.position='none') + 
  labs(x ="Hidden Layer", y = "TF-XL Similarity")+ scale_fill_manual(values = c("#9999CC", "gold3", "darkorchid3", "darkgoldenrod4"), name= "Noun+Verb", labels=c("High+NonIC", "Low+NonIC", "High+IC", "Low+IC"))


#GPT-2 XL who
gpt_who_sim$hasIC <- factor(gpt_who_sim$hasIC)
gpt_who_sim <- subset(gpt_who_sim, num=='pl')
gpt_who_sim$NP <- factor(gpt_who_sim$NP)
gpt_who_sim_small <- subset(gpt_who_sim, layer > 29)

gpt_who <- ggplot(gpt_who_sim_small, aes(x=factor(layer), y=sim, fill=interaction(NP, hasIC))) +
  geom_boxplot(notch=TRUE) + ylim(0, 1) + theme(text = element_text(size=12)) + theme(legend.position='none') + 
  labs(x ="Hidden Layer", y = "GPT-2 XL Similarity") + scale_fill_manual(values = c("#9999CC", "gold3", "darkorchid3", "darkgoldenrod4"), name= "Noun+Verb", labels=c("High+NonIC", "Low+NonIC", "High+IC", "Low+IC"))

plot_row <- plot_grid(tf_who, lstm_who, rel_widths =c(1.8, 1))

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
  draw_plot(gpt_who, x=0, y= 0, width=1, height=0.5) + 
  draw_plot_label(label = c("c)", "a)", "b)"), size = 15,
                  x = c(0.03, 0.03, 0.65), y = c(0.07, 0.57, 0.57))


#TF-XL were
tf_were_HIGH <- subset(tf_were_sim, isHIGH==1)
tf_were_LOW <- subset(tf_were_sim, isHIGH==0)

tf_were_h <- ggplot(tf_were_HIGH, aes(x=factor(layer), y=sim, fill=interaction(NP, hasIC))) +
  geom_boxplot(notch=TRUE)  + ylim(-0.1, 1)+ theme(text = element_text(size=12))  + 
  labs(x ="Hidden Layer", y = "TF-XL Similarity with High Agree") + scale_fill_manual(values = c("#9999CC", "gold3", "darkorchid3", "darkgoldenrod4"), name= "Noun+Verb", labels=c("High+NonIC", "Low+NonIC", "High+IC", "Low+IC"))

tf_were_h <- tf_were_h  + theme(legend.position = c(0.05, 0.95), 
                              legend.justification = c("left", "top"), 
                              legend.box.just = "right", 
                              legend.margin = margin(6, 6, 6, 6))


tf_were_l <- ggplot(tf_were_LOW, aes(x=factor(layer), y=sim, fill=interaction(NP, hasIC))) +
  geom_boxplot(notch=TRUE)  + ylim(-0.1, 1)+ theme(text = element_text(size=12))  + theme(legend.position = 'none') + 
  labs(x ="Hidden Layer", y = "TF-XL Similarity with Low Agree") + scale_fill_manual(values = c("#9999CC", "gold3", "darkorchid3", "darkgoldenrod4"), name= "Noun+Verb", labels=c("High+NonIC", "Low+NonIC", "High+IC", "Low+IC"))

  
  tf_were_plots <- plot_grid(tf_were_h, tf_were_l, ncol=1, labels = c('a)', 'b)'), 
                             label_size = 12,
                             label_x = 0.03, label_y = 0.01,
                             hjust = -0.5, vjust = -0.5)
  
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
  tf_were_plots <- plot_grid(
    title, tf_were_plots,
    ncol = 1,
    # rel_heights values control vertical title margins
    rel_heights = c(0.1, 1.2)
  )

 
#GPT-2 XL were
gpt_were_HIGH <- subset(gpt_were_sim, isHIGH==1)
gpt_were_LOW <- subset(gpt_were_sim, isHIGH==0)

gpt_were_HIGH_small <- subset(gpt_were_HIGH, layer>24)
gpt_were_LOW_small <- subset(gpt_were_LOW, layer>24)

gpt_were_h <- ggplot(gpt_were_HIGH_small, aes(x=factor(layer), y=sim, fill=interaction(NP, hasIC))) +
geom_boxplot(notch=TRUE)  + ylim(-0.1, 1)+ theme(text = element_text(size=12))  + 
labs(x ="Hidden Layer", y = "GPT-2 XL Similarity with High Agree") + scale_fill_manual(values = c("#9999CC", "gold3", "darkorchid3", "darkgoldenrod4"), name= "Noun+Verb", labels=c("High+NonIC", "Low+NonIC", "High+IC", "Low+IC"))

gpt_were_h <- gpt_were_h  + theme(legend.position = c(0.01, 0.01), 
                              legend.justification = c("left", "bottom"), 
                              legend.box.just = "right", 
                              legend.margin = margin(6, 6, 6, 6))


gpt_were_l <- ggplot(gpt_were_LOW_small, aes(x=factor(layer), y=sim, fill=interaction(NP, hasIC))) +
geom_boxplot(notch=TRUE)  + ylim(-0.1, 1)+ theme(text = element_text(size=12))  + theme(legend.position = 'none') + 
labs(x ="Hidden Layer", y = "GPT-2 XL Similarity with Low Agree") + scale_fill_manual(values = c("#9999CC", "gold3", "darkorchid3", "darkgoldenrod4"), name= "Noun+Verb", labels=c("High+NonIC", "Low+NonIC", "High+IC", "Low+IC"))


gpt_were_plots <- plot_grid(gpt_were_h, gpt_were_l, ncol=1, labels = c('a)', 'b)'), 
                         label_size = 12,
                         label_x = 0.03, label_y = 0.01,
                         hjust = -0.5, vjust = -0.5)

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
gpt_were_plots <- plot_grid(
title, gpt_were_plots,
ncol = 1,
# rel_heights values control vertical title margins
rel_heights = c(0.1, 1.2)
) 


#########################
# Reading Time were SIM Stats
#########################

rt_sim_lstm_model <- lmer(sim ~ hasIC*NP*layer*isHIGH + (1|item), data=lstm_were_sim)
summary(rt_sim_lstm_model)
anova(rt_sim_lstm_model)

rt_sim_tf_model <- lmer(sim ~ hasIC*NP*layer*isHIGH + (1|item), data=tf_were_sim)
summary(rt_sim_tf_model)
anova(rt_sim_tf_model)

rt_sim_gpt_model <- lmer(sim ~ hasIC*NP*layer*isHIGH + (1|item), data=gpt_were_sim)
summary(rt_sim_gpt_model)
anova(rt_sim_gpt_model)

#########################
# Pronoun Surp Stats
#########################

pronoun_surp$hasIC <- pronoun_surp$bias > 0
pronoun_surp$hasIC <- as.numeric(pronoun_surp$hasIC)

lstm_pronoun_model <- lmer(LSTM_avg_surp ~ hasIC*isHigh*gender + (1|item), data=pronoun_surp)
summary(lstm_pronoun_model)
anova(lstm_pronoun_model)

tf_pronoun_model <- lmer(tf_surp ~ hasIC*isHigh*gender + (1|item), data=pronoun_surp)
summary(tf_pronoun_model)
anova(tf_pronoun_model)

gpt_pronoun_model <- lmer(gpt_surp ~ hasIC*isHigh*gender + (1|item), data=pronoun_surp)
summary(gpt_pronoun_model)
anova(gpt_pronoun_model)

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

# LSTM
lstm_pronoun <- ggplot(pronoun_surp, aes(x=factor(hasIC), y=LSTM_avg_surp, fill=interaction(factor(gender), factor(isHigh)))) +
  geom_boxplot(notch=TRUE) + ylim(0, 11)+ theme(text = element_text(size=12)) + 
  labs(x ="Verb Type", y = "LSTM Surprisal (Pronoun)") + 
  scale_x_discrete(breaks=c("0","1"),
                   labels=c("Object-Bias", "Subject-Bias")) + 
  scale_fill_manual(values = c("#9999CC", "gold3", "darkorchid3", "darkgoldenrod4"), name= "Gender+Type", labels=c("f+Obj", "m+Obj", "f+Subj", "m+Subj"))

legend <- get_legend(lstm_pronoun)

lstm_pronoun <- lstm_pronoun + theme(legend.position='none')

# GPT-2 XL
gpt_pronoun <- ggplot(pronoun_surp, aes(x=factor(hasIC), y=gpt_surp, fill=interaction(factor(gender), factor(isHigh)))) +
  geom_boxplot(notch=TRUE) + ylim(0, 11)+ theme(text = element_text(size=12)) + 
  labs(x ="Verb Type", y = "GPT-2 XL Surprisal (Pronoun)") + theme(legend.position = 'none') +
  scale_x_discrete(breaks=c("0","1"),
                   labels=c("Object-Bias", "Subject-Bias")) + 
  scale_fill_manual(values = c("#9999CC", "gold3", "darkorchid3", "darkgoldenrod4"), name= "Gender+Type", labels=c("f+Obj", "m+Obj", "f+Subj", "m+Subj"))

# TF-XL
tf_pronoun <- ggplot(pronoun_surp, aes(x=factor(hasIC), y=tf_surp, fill=interaction(factor(gender), factor(isHigh)))) +
  geom_boxplot(notch=TRUE) + ylim(0, 11)+ theme(text = element_text(size=12)) + 
  labs(x ="Verb Type", y = "TF-XL Surprisal (Pronoun)") + theme(legend.position = 'none') +
  scale_x_discrete(breaks=c("0","1"),
                   labels=c("Object-Bias", "Subject-Bias")) + 
  scale_fill_manual(values = c("#9999CC", "gold3", "darkorchid3", "darkgoldenrod4"), name= "Gender+Type", labels=c("f+Obj", "m+Obj", "f+Subj", "m+Subj"))

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
    plot.margin = margin(0, 0, 0, 7)
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
  draw_plot(legend, x=0.5, y= 0, width=0.5, height=0.5) + 
  draw_plot_label(label = c("c)", "a)", "b)"), size = 15,
                  x = c(0.05, 0.05, 0.55), y = c(0.07, 0.57, 0.57))

#########################
# Pronoun SIM Stats
#########################
lstm_pronoun_sim$hasIC <- lstm_pronoun_sim$bias > 0
lstm_pronoun_sim$hasIC <- as.numeric(lstm_pronoun_sim$hasIC)

pronoun_sim_lstm_model <- lmer(sim ~ hasIC*NP*layer*gender + (1|item), data=lstm_pronoun_sim)
summary(pronoun_sim_lstm_model)
anova(pronoun_sim_lstm_model)

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

tf_pronoun_HIGH_IC <- subset(tf_pronoun_sim, hasIC==1 & layer == 18 & NP == 1)
tf_pronoun_HIGH_nonIC <- subset(tf_pronoun_sim, hasIC==0 & layer == 18 & NP == 1)

tf_pronoun_LOW_IC <- subset(tf_pronoun_sim, hasIC==1 & layer == 18 & NP == 2)
tf_pronoun_LOW_nonIC <- subset(tf_pronoun_sim, hasIC==0 & layer == 18 & NP == 2)

t.test(tf_pronoun_HIGH_IC$sim, tf_pronoun_HIGH_nonIC$sim)
t.test(tf_pronoun_LOW_IC$sim, tf_pronoun_LOW_nonIC$sim)


gpt_pronoun_sim$hasIC <- gpt_pronoun_sim$bias > 0
gpt_pronoun_sim$hasIC <- as.numeric(gpt_pronoun_sim$hasIC)

pronoun_sim_gpt_model <- lmer(sim ~ hasIC*NP*layer*gender + (1|item), data=gpt_pronoun_sim)
summary(pronoun_sim_gpt_model)
anova(pronoun_sim_gpt_model)

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
  geom_boxplot(notch=TRUE) + ylim(0, 1) + theme(text = element_text(size=12)) + 
  labs(x ="Hidden Layer", y = "LSTM Similarity") + 
  scale_fill_manual(values = c("#9999CC", "gold3", "darkorchid3", "darkgoldenrod4"), name= "Noun+Bias", labels=c("Subj+Obj Bias", "Obj+Obj Bias", "Subj+Subj Bias", "Obj+Subj Bias"))

lstm_pronoun <- lstm_pronoun  + theme(legend.position = c(0.0, 0.96), 
                              legend.justification = c("left", "top"), 
                              legend.box.just = "right", 
                              legend.margin = margin(6, 6, 6, 6))

#TF-XL who
tf_pronoun_sim$hasIC <- factor(tf_pronoun_sim$hasIC)
tf_pronoun_sim$NP <- factor(tf_pronoun_sim$NP)

tf_pronoun <- ggplot(tf_pronoun_sim, aes(x=factor(layer), y=sim, fill=interaction(NP, hasIC))) +
  geom_boxplot(notch=TRUE) + ylim(0, 1) + theme(text = element_text(size=12)) + theme(legend.position='none') + 
  labs(x ="Hidden Layer", y = "TF-XL Similarity") + 
  scale_fill_manual(values = c("#9999CC", "gold3", "darkorchid3", "darkgoldenrod4"), name= "Noun+Bias", labels=c("Subj+Obj Bias", "Obj+Obj Bias", "Subj+Subj Bias", "Obj+Subj Bias"))


#GPT-2 XL who
gpt_pronoun_sim$hasIC <- factor(gpt_pronoun_sim$hasIC)
gpt_pronoun_sim$NP <- factor(gpt_pronoun_sim$NP)
gpt_pronoun_sim_small <- subset(gpt_pronoun_sim, layer > 29)

gpt_pronoun <- ggplot(gpt_pronoun_sim_small, aes(x=factor(layer), y=sim, fill=interaction(NP, hasIC))) +
  geom_boxplot(notch=TRUE) + ylim(0, 1) + theme(text = element_text(size=12)) + theme(legend.position='none') + 
  labs(x ="Hidden Layer", y = "GPT-2 XL Similarity") + 
  scale_fill_manual(values = c("#9999CC", "gold3", "darkorchid3", "darkgoldenrod4"), name= "Noun+Bias", labels=c("Subj+Obj Bias", "Obj+Obj Bias", "Subj+Subj Bias", "Obj+Subj Bias"))


plot_row <- plot_grid(tf_pronoun, lstm_pronoun, rel_widths =c(1.8, 1))

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
  draw_plot(gpt_pronoun, x=0, y= 0, width=1, height=0.5) + 
  draw_plot_label(label = c("c)", "a)", "b)"), size = 15,
                  x = c(0.03, 0.03, 0.65), y = c(0.07, 0.57, 0.57))

