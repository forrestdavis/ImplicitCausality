library(ggplot2)
###########################
###### LOAD DATA ##########
###########################
path = "/home/forrestdavis/Projects/ImplicitCausality/"

bert_en_score <- read.csv(paste(path, "results/IC_mismatch_EN.csv", sep=''))
bert_es_score <- read.csv(paste(path, "results/../IC_mismatch_ES.csv", sep=''))
bert_it_score <- read.csv(paste(path, "results/../IC_mismatch_IT.csv", sep=''))
bert_zh_score <- read.csv(paste(path, "results/IC_mismatch_ZH.csv", sep=''))

bert_en_score$hasIC <- as.numeric(bert_en_score$bias>0)
bert_en_score$isHigh <- factor(bert_en_score$isHigh)
bert_en_score$gender <- factor(bert_en_score$gender)
bert_en_score$hasIC <- factor(bert_en_score$hasIC)

bert_es_score$hasIC <- as.numeric(bert_es_score$bias>50)
bert_es_score$isHigh <- factor(bert_es_score$isHigh)
bert_es_score$gender <- factor(bert_es_score$gender)
bert_es_score$hasIC <- factor(bert_es_score$hasIC)

bert_it_score$hasIC <- as.numeric(bert_it_score$bias>50)
bert_it_score$isHigh <- factor(bert_it_score$isHigh)
bert_it_score$gender <- factor(bert_it_score$gender)
bert_it_score$hasIC <- factor(bert_it_score$hasIC)

bert_zh_score$hasIC <- as.numeric(bert_zh_score$bias>50)
bert_zh_score$isHigh <- factor(bert_zh_score$isHigh)
bert_zh_score$gender <- factor(bert_zh_score$gender)
bert_zh_score$hasIC <- factor(bert_zh_score$hasIC)


###########################
######  PLOT EN/ZH   ######
###########################

bert_en_pronoun <- ggplot(bert_en_score, aes(x=hasIC, y=score_bert, fill=isHigh)) +
  geom_boxplot(notch=TRUE, outlier.size = 0.1) + ylim(0, 1)+ theme(text = element_text(size=22)) +
   labs(x ="Verb Bias", y = "BERT Probability") +
  scale_x_discrete(breaks=c("0","1"),
                   labels=c("Object-Bias", "Subject-Bias")) +
  scale_fill_manual(values = c("#9999CC", "darkorchid3"), name= "Antecedent", labels=c("Object", "Subject"))

bert_en_pronoun <- bert_en_pronoun + theme(legend.position='none')

roberta_en_pronoun <- ggplot(bert_en_score, aes(x=hasIC, y=score_roberta, fill=isHigh)) +
  geom_boxplot(notch=TRUE, outlier.size = 0.1) + ylim(0, 1)+ theme(text = element_text(size=22)) + 
  labs(x ="Verb Bias", y = "RoBERTa Score") +
  scale_x_discrete(breaks=c("0","1"),
                   labels=c("Object-Bias", "Subject-Bias")) + 
  scale_fill_manual(values = c("#9999CC", "darkorchid3"), name= "Antecedent", labels=c("Object", "Subject"))

roberta_en_pronoun <- roberta_en_pronoun + theme(legend.position='none')

bert_zh_pronoun <- ggplot(bert_zh_score, aes(x=hasIC, y=score_bert, fill=isHigh)) +
  geom_boxplot(notch=TRUE, outlier.size = 0.1) + ylim(0, 1)+ theme(text = element_text(size=22)) + 
  labs(x ="Verb Bias", y = "Chinese BERT Score") +
  scale_x_discrete(breaks=c("0","1"),
                   labels=c("Object-Bias", "Subject-Bias")) + 
  scale_fill_manual(values = c("#9999CC", "darkorchid3"), name= "Antecedent", labels=c("Object", "Subject"))

bert_zh_pronoun <- bert_zh_pronoun + theme(legend.position='none')

roberta_zh_pronoun <- ggplot(bert_zh_score, aes(x=hasIC, y=score_roberta, fill=isHigh)) +
  geom_boxplot(notch=TRUE, outlier.size = 0.1) + ylim(0, 1)+ theme(text = element_text(size=22)) + 
  labs(x ="Verb Bias", y = "Chinese RoBERTa Score") +
  scale_x_discrete(breaks=c("0","1"),
                   labels=c("Object-Bias", "Subject-Bias")) + 
  scale_fill_manual(values = c("#9999CC", "darkorchid3"), name= "Antecedent", labels=c("Object", "Subject"))

roberta_zh_pronoun <- roberta_zh_pronoun + theme(legend.text = element_text(size=28),
      legend.title = element_text(size=28),
      legend.position = c(0.45, 0.98), 
      legend.justification = c("left", "top"), 
      legend.box.just = "center", 
      legend.margin = margin(6, 6, 6, 6))

plot_row <- plot_grid(bert_en_pronoun, roberta_en_pronoun)

title <- ggdraw() + 
  draw_label(
    "(Ro)BERT(a) Pronoun Score",
    fontface = 'bold',
    x = 0,
    size=28,
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
og_plots <- ggdraw() + draw_plot(plot_row, x=0, y= 0.5, width=1, height=0.5) +
  draw_plot(bert_zh_pronoun, x=0, y= 0, width=0.5, height=0.5) + 
  draw_plot(roberta_zh_pronoun, x=0.5, y= 0, width=0.5, height=0.5) + 
  draw_plot_label(label = c("c)", "a)", "b)", "d)"), size = 15,
                  x = c(0.05, 0.05, 0.55, 0.55), y = c(0.07, 0.57, 0.57, 0.07))


###########################
#### PLOT EN/ZH BASE ######
###########################

bert_en_pronoun <- ggplot(bert_en_score, aes(x=hasIC, y=score_bert_base, fill=isHigh)) +
  geom_boxplot(notch=TRUE, outlier.size = 0.1) + ylim(0, 1)+ theme(text = element_text(size=22)) + 
  labs(x ="Verb Bias", y = "BERT Score") +
  scale_x_discrete(breaks=c("0","1"),
                   labels=c("Object-Bias", "Subject-Bias")) + 
  scale_fill_manual(values = c("#9999CC", "darkorchid3"), name= "Antecedent", labels=c("Object", "Subject"))

bert_en_pronoun <- bert_en_pronoun + theme(legend.position='none')

roberta_en_pronoun <- ggplot(bert_en_score, aes(x=hasIC, y=score_roberta_base, fill=isHigh)) +
  geom_boxplot(notch=TRUE, outlier.size = 0.1) + ylim(0, 1)+ theme(text = element_text(size=22)) + 
  labs(x ="Verb Bias", y = "RoBERTa Score") +
  scale_x_discrete(breaks=c("0","1"),
                   labels=c("Object-Bias", "Subject-Bias")) + 
  scale_fill_manual(values = c("#9999CC", "darkorchid3"), name= "Antecedent", labels=c("Object", "Subject"))

roberta_en_pronoun <- roberta_en_pronoun + theme(legend.position='none')

bert_zh_pronoun <- ggplot(bert_zh_score, aes(x=hasIC, y=score_bert_base, fill=isHigh)) +
  geom_boxplot(notch=TRUE, outlier.size = 0.1) + ylim(0, 1)+ theme(text = element_text(size=22)) + 
  labs(x ="Verb Bias", y = "Chinese BERT Score") +
  scale_x_discrete(breaks=c("0","1"),
                   labels=c("Object-Bias", "Subject-Bias")) + 
  scale_fill_manual(values = c("#9999CC", "darkorchid3"), name= "Antecedent", labels=c("Object", "Subject"))

bert_zh_pronoun <- bert_zh_pronoun + theme(legend.position='none')

roberta_zh_pronoun <- ggplot(bert_zh_score, aes(x=hasIC, y=score_roberta_base, fill=isHigh)) +
  geom_boxplot(notch=TRUE, outlier.size = 0.1) + ylim(0, 1)+ theme(text = element_text(size=22)) + 
  labs(x ="Verb Bias", y = "Chinese RoBERTa Score") +
  scale_x_discrete(breaks=c("0","1"),
                   labels=c("Object-Bias", "Subject-Bias")) + 
  scale_fill_manual(values = c("#9999CC", "darkorchid3"), name= "Antecedent", labels=c("Object", "Subject"))

roberta_zh_pronoun <- roberta_zh_pronoun + theme(legend.text = element_text(size=28),
                                                 legend.title = element_text(size=28),
                                                 legend.position = c(0.45, 0.98), 
                                                 legend.justification = c("left", "top"), 
                                                 legend.box.just = "center", 
                                                 legend.margin = margin(6, 6, 6, 6))

plot_row <- plot_grid(bert_en_pronoun, roberta_en_pronoun)

title <- ggdraw() + 
  draw_label(
    "(Ro)BERT(a) Fine-Tuned Baseline Pronoun Score",
    fontface = 'bold',
    x = 0,
    size=28,
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
base_plots <- ggdraw() + draw_plot(plot_row, x=0, y= 0.5, width=1, height=0.5) +
  draw_plot(bert_zh_pronoun, x=0, y= 0, width=0.5, height=0.5) + 
  draw_plot(roberta_zh_pronoun, x=0.5, y= 0, width=0.5, height=0.5) + 
  draw_plot_label(label = c("c)", "a)", "b)", "d)"), size = 15,
                  x = c(0.05, 0.05, 0.55, 0.55), y = c(0.07, 0.57, 0.57, 0.07))


###########################
#### PLOT EN/ZH Pro  ######
###########################

bert_en_pronoun <- ggplot(bert_en_score, aes(x=hasIC, y=score_bert_pro, fill=isHigh)) +
  geom_boxplot(notch=TRUE, outlier.size = 0.1) + ylim(0, 1)+ theme(text = element_text(size=22)) + 
  labs(x ="Verb Bias", y = "BERT Score") +
  scale_x_discrete(breaks=c("0","1"),
                   labels=c("Object-Bias", "Subject-Bias")) + 
  scale_fill_manual(values = c("#9999CC", "darkorchid3"), name= "Antecedent", labels=c("Object", "Subject"))

bert_en_pronoun <- bert_en_pronoun + theme(legend.position='none')

roberta_en_pronoun <- ggplot(bert_en_score, aes(x=hasIC, y=score_roberta_pro, fill=isHigh)) +
  geom_boxplot(notch=TRUE, outlier.size = 0.1) + ylim(0, 1)+ theme(text = element_text(size=22)) + 
  labs(x ="Verb Bias", y = "RoBERTa Score") +
  scale_x_discrete(breaks=c("0","1"),
                   labels=c("Object-Bias", "Subject-Bias")) + 
  scale_fill_manual(values = c("#9999CC", "darkorchid3"), name= "Antecedent", labels=c("Object", "Subject"))

roberta_en_pronoun <- roberta_en_pronoun + theme(legend.position='none')

bert_zh_pronoun <- ggplot(bert_zh_score, aes(x=hasIC, y=score_bert_pro, fill=isHigh)) +
  geom_boxplot(notch=TRUE, outlier.size = 0.1) + ylim(0, 1)+ theme(text = element_text(size=22)) + 
  labs(x ="Verb Bias", y = "Chinese BERT Score") +
  scale_x_discrete(breaks=c("0","1"),
                   labels=c("Object-Bias", "Subject-Bias")) + 
  scale_fill_manual(values = c("#9999CC", "darkorchid3"), name= "Antecedent", labels=c("Object", "Subject"))

bert_zh_pronoun <- bert_zh_pronoun + theme(legend.position='none')

roberta_zh_pronoun <- ggplot(bert_zh_score, aes(x=hasIC, y=score_roberta_pro, fill=isHigh)) +
  geom_boxplot(notch=TRUE, outlier.size = 0.1) + ylim(0, 1)+ theme(text = element_text(size=22)) + 
  labs(x ="Verb Bias", y = "Chinese RoBERTa Score") +
  scale_x_discrete(breaks=c("0","1"),
                   labels=c("Object-Bias", "Subject-Bias")) + 
  scale_fill_manual(values = c("#9999CC", "darkorchid3"), name= "Antecedent", labels=c("Object", "Subject"))

roberta_zh_pronoun <- roberta_zh_pronoun + theme(legend.text = element_text(size=28),
                                                 legend.title = element_text(size=28),
                                                 legend.position = c(0.45, 0.98), 
                                                 legend.justification = c("left", "top"), 
                                                 legend.box.just = "center", 
                                                 legend.margin = margin(6, 6, 6, 6))

plot_row <- plot_grid(bert_en_pronoun, roberta_en_pronoun)

title <- ggdraw() + 
  draw_label(
    "(Ro)BERT(a) Fine-Tuned with Pro Drop Pronoun Score",
    fontface = 'bold',
    x = 0,
    size=28,
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
pro_plots <- ggdraw() + draw_plot(plot_row, x=0, y= 0.5, width=1, height=0.5) +
  draw_plot(bert_zh_pronoun, x=0, y= 0, width=0.5, height=0.5) + 
  draw_plot(roberta_zh_pronoun, x=0.5, y= 0, width=0.5, height=0.5) + 
  draw_plot_label(label = c("c)", "a)", "b)", "d)"), size = 15,
                  x = c(0.05, 0.05, 0.55, 0.55), y = c(0.07, 0.57, 0.57, 0.07))



###########################
######  PLOT ES/IT   ######
###########################

bert_es_pronoun <- ggplot(bert_es_score, aes(x=hasIC, y=sad, fill=isHigh)) +
  geom_boxplot(notch=TRUE, outlier.size = 0.1) + ylim(0, 1)+ theme(text = element_text(size=22)) + 
  labs(x ="Verb Bias", y = "Spanish BERT Probability") +
  scale_x_discrete(breaks=c("0","1"),
                   labels=c("Object-Bias", "Subject-Bias")) + 
  scale_fill_manual(values = c("#9999CC", "darkorchid3"), name= "Antecedent", labels=c("Object", "Subject"))

bert_es_pronoun

bert_es_pronoun <- bert_es_pronoun + theme(legend.position='none')

bert_it_pronoun <- ggplot(bert_it_score, aes(x=hasIC, y=score, fill=isHigh)) +
  geom_boxplot(notch=TRUE, outlier.size = 0.1) + ylim(0, 1)+ theme(text = element_text(size=22)) + 
  labs(x ="Verb Bias", y = "Italian BERT Probability") +
  scale_x_discrete(breaks=c("0","1"),
                   labels=c("Object-Bias", "Subject-Bias")) + 
  scale_fill_manual(values = c("#9999CC", "darkorchid3"), name= "Antecedent", labels=c("Object", "Subject"))

bert_it_pronoun <- bert_it_pronoun + theme(legend.position='none')

umberto_it_pronoun <- ggplot(bert_it_score, aes(x=hasIC, y=score_add, fill=isHigh)) +
  geom_boxplot(notch=TRUE, outlier.size = 0.1) + ylim(0, 1)+ theme(text = element_text(size=22)) + 
  labs(x ="Verb Bias", y = "Italian UmBERTo Score") +
  scale_x_discrete(breaks=c("0","1"),
                   labels=c("Object-Bias", "Subject-Bias")) + 
  scale_fill_manual(values = c("#9999CC", "darkorchid3"), name= "Antecedent", labels=c("Object", "Subject"))

umberto_it_pronoun <- umberto_it_pronoun + theme(legend.position='none')

gilberto_it_pronoun <- ggplot(bert_it_score, aes(x=hasIC, y=score_gil, fill=isHigh)) +
  geom_boxplot(notch=TRUE, outlier.size = 0.1) + ylim(0, 1)+ theme(text = element_text(size=22)) + 
  labs(x ="Verb Bias", y = "Italian GilBERTo Score") +
  scale_x_discrete(breaks=c("0","1"),
                   labels=c("Object-Bias", "Subject-Bias")) + 
  scale_fill_manual(values = c("#9999CC", "darkorchid3"), name= "Antecedent", labels=c("Object", "Subject"))

gilberto_it_pronoun <- gilberto_it_pronoun + theme(legend.text = element_text(size=28),
                                                 legend.title = element_text(size=28),
                                                 legend.position = c(0.45, 0.98), 
                                                 legend.justification = c("left", "top"), 
                                                 legend.box.just = "center", 
                                                 legend.margin = margin(6, 6, 6, 6))

plot_row <- plot_grid(bert_es_pronoun, bert_it_pronoun)

title <- ggdraw() + 
  draw_label(
    "(Ro)BERT(a) Pronoun Score",
    fontface = 'bold',
    x = 0,
    size=28,
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
og_plots <- ggdraw() + draw_plot(plot_row, x=0, y= 0.5, width=1, height=0.5) +
  draw_plot(umberto_it_pronoun, x=0, y= 0, width=0.5, height=0.5) + 
  draw_plot(gilberto_it_pronoun, x=0.5, y= 0, width=0.5, height=0.5) + 
  draw_plot_label(label = c("c)", "a)", "b)", "d)"), size = 15,
                  x = c(0.05, 0.05, 0.55, 0.55), y = c(0.07, 0.57, 0.57, 0.07))



###########################
#### PLOT ES/IT BASE ######
###########################

bert_es_pronoun <- ggplot(bert_es_score, aes(x=hasIC, y=score_base, fill=isHigh)) +
  geom_boxplot(notch=TRUE, outlier.size = 0.1) + ylim(0, 1)+ theme(text = element_text(size=22)) + 
  labs(x ="Verb Bias", y = "Spanish BERT Probability", title='Fine-tuned on unmodified sentences') +
  scale_x_discrete(breaks=c("0","1"),
                   labels=c("Object-Bias", "Subject-Bias")) + 
  scale_fill_manual(values = c("#9999CC", "darkorchid3"), name= "Antecedent", labels=c("Object", "Subject"))

bert_es_pronoun <- bert_es_pronoun + theme(legend.position='none')

bert_it_pronoun <- ggplot(bert_it_score, aes(x=hasIC, y=score_base, fill=isHigh)) +
  geom_boxplot(notch=TRUE, outlier.size = 0.1) + ylim(0, 1)+ theme(text = element_text(size=22)) + 
  labs(x ="Verb Bias", y = "Italian BERT Probability", title='Fine-tuned on unmodified sentences') +
  scale_x_discrete(breaks=c("0","1"),
                   labels=c("Object-Bias", "Subject-Bias")) + 
  scale_fill_manual(values = c("#9999CC", "darkorchid3"), name= "Antecedent", labels=c("Object", "Subject"))

bert_it_pronoun <- bert_it_pronoun + theme(legend.position='none')

umberto_it_pronoun <- ggplot(bert_it_score, aes(x=hasIC, y=score_um_base, fill=isHigh)) +
  geom_boxplot(notch=TRUE, outlier.size = 0.1) + ylim(0, 1)+ theme(text = element_text(size=22)) + 
  labs(x ="Verb Bias", y = "Italian UmBERTo Score") +
  scale_x_discrete(breaks=c("0","1"),
                   labels=c("Object-Bias", "Subject-Bias")) + 
  scale_fill_manual(values = c("#9999CC", "darkorchid3"), name= "Antecedent", labels=c("Object", "Subject"))

umberto_it_pronoun <- umberto_it_pronoun + theme(legend.position='none')

gilberto_it_pronoun <- ggplot(bert_it_score, aes(x=hasIC, y=score_gil_base, fill=isHigh)) +
  geom_boxplot(notch=TRUE, outlier.size = 0.1) + ylim(0, 1)+ theme(text = element_text(size=22)) + 
  labs(x ="Verb Bias", y = "Italian GilBERTo Score") +
  scale_x_discrete(breaks=c("0","1"),
                   labels=c("Object-Bias", "Subject-Bias")) + 
  scale_fill_manual(values = c("#9999CC", "darkorchid3"), name= "Antecedent", labels=c("Object", "Subject"))

gilberto_it_pronoun <- gilberto_it_pronoun + theme(legend.text = element_text(size=28),
                                                   legend.title = element_text(size=28),
                                                   legend.position = c(0.45, 0.98), 
                                                   legend.justification = c("left", "top"), 
                                                   legend.box.just = "center", 
                                                   legend.margin = margin(6, 6, 6, 6))

plot_row <- plot_grid(bert_es_pronoun, bert_it_pronoun)

title <- ggdraw() + 
  draw_label(
    "(Ro)BERT(a) Fine-Tuned Baseline Pronoun Score",
    fontface = 'bold',
    x = 0,
    size=28,
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
og_plots <- ggdraw() + draw_plot(plot_row, x=0, y= 0.5, width=1, height=0.5) +
  draw_plot(umberto_it_pronoun, x=0, y= 0, width=0.5, height=0.5) + 
  draw_plot(gilberto_it_pronoun, x=0.5, y= 0, width=0.5, height=0.5) + 
  draw_plot_label(label = c("c)", "a)", "b)", "d)"), size = 15,
                  x = c(0.05, 0.05, 0.55, 0.55), y = c(0.07, 0.57, 0.57, 0.07))



###########################
#### PLOT ES/IT Pro  ######
###########################

bert_es_pronoun <- ggplot(bert_es_score, aes(x=hasIC, y=score_pro, fill=isHigh)) +
  geom_boxplot(notch=TRUE, outlier.size = 0.1) + ylim(0, 1)+ theme(text = element_text(size=22)) + 
  labs(x ="Verb Bias", y = "Spanish BERT Probability", title = "Fine-tuned without ProDrop") +
  scale_x_discrete(breaks=c("0","1"),
                   labels=c("Object-Bias", "Subject-Bias")) + 
  scale_fill_manual(values = c("#9999CC", "darkorchid3"), name= "Antecedent", labels=c("Object", "Subject"))

bert_es_pronoun <- bert_es_pronoun + theme(legend.position='none')

bert_it_pronoun <- ggplot(bert_it_score, aes(x=hasIC, y=score_pro, fill=isHigh)) +
  geom_boxplot(notch=TRUE, outlier.size = 0.1) + ylim(0, 1)+ theme(text = element_text(size=22)) + 
  labs(x ="Verb Bias", y = "Italian BERT Probability", title="Fine-tuned without ProDrop") +
  scale_x_discrete(breaks=c("0","1"),
                   labels=c("Object-Bias", "Subject-Bias")) + 
  scale_fill_manual(values = c("#9999CC", "darkorchid3"), name= "Antecedent", labels=c("Object", "Subject"))

bert_it_pronoun <- bert_it_pronoun + theme(legend.position='none')

umberto_it_pronoun <- ggplot(bert_it_score, aes(x=hasIC, y=score_um_pro, fill=isHigh)) +
  geom_boxplot(notch=TRUE, outlier.size = 0.1) + ylim(0, 1)+ theme(text = element_text(size=22)) + 
  labs(x ="Verb Bias", y = "Italian UmBERTo Score") +
  scale_x_discrete(breaks=c("0","1"),
                   labels=c("Object-Bias", "Subject-Bias")) + 
  scale_fill_manual(values = c("#9999CC", "darkorchid3"), name= "Antecedent", labels=c("Object", "Subject"))

umberto_it_pronoun <- umberto_it_pronoun + theme(legend.position='none')

gilberto_it_pronoun <- ggplot(bert_it_score, aes(x=hasIC, y=score_gil_pro, fill=isHigh)) +
  geom_boxplot(notch=TRUE, outlier.size = 0.1) + ylim(0, 1)+ theme(text = element_text(size=22)) + 
  labs(x ="Verb Bias", y = "Italian GilBERTo Score") +
  scale_x_discrete(breaks=c("0","1"),
                   labels=c("Object-Bias", "Subject-Bias")) + 
  scale_fill_manual(values = c("#9999CC", "darkorchid3"), name= "Antecedent", labels=c("Object", "Subject"))

gilberto_it_pronoun <- gilberto_it_pronoun + theme(legend.text = element_text(size=28),
                                                   legend.title = element_text(size=28),
                                                   legend.position = c(0.45, 0.98), 
                                                   legend.justification = c("left", "top"), 
                                                   legend.box.just = "center", 
                                                   legend.margin = margin(6, 6, 6, 6))

plot_row <- plot_grid(bert_es_pronoun, bert_it_pronoun)

title <- ggdraw() + 
  draw_label(
    "(Ro)BERT(a) Fine-Tuned no Pro Drop Pronoun Score",
    fontface = 'bold',
    x = 0,
    size=28,
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
og_plots <- ggdraw() + draw_plot(plot_row, x=0, y= 0.5, width=1, height=0.5) +
  draw_plot(umberto_it_pronoun, x=0, y= 0, width=0.5, height=0.5) + 
  draw_plot(gilberto_it_pronoun, x=0.5, y= 0, width=0.5, height=0.5) + 
  draw_plot_label(label = c("c)", "a)", "b)", "d)"), size = 15,
                  x = c(0.05, 0.05, 0.55, 0.55), y = c(0.07, 0.57, 0.57, 0.07))



mbert_pronoun <- ggplot(bert_en_score, aes(x=hasIC, y=score_mbert, fill=isHigh)) +
  geom_boxplot(notch=TRUE, outlier.size = 0.1) + ylim(0, 1)+ theme(text = element_text(size=22)) + 
  labs(x ="Verb Bias", y = "mBERT Probability (English)") +
  scale_x_discrete(breaks=c("0","1"),
                   labels=c("Object-Bias", "Subject-Bias")) + 
  scale_fill_manual(values = c("#9999CC", "darkorchid3"), name= "Antecedent", labels=c("Object", "Subject"))

mbert_pronoun

mbert_pronoun <- ggplot(bert_es_score, aes(x=hasIC, y=score_mbert, fill=isHigh)) +
  geom_boxplot(notch=TRUE, outlier.size = 0.1) + ylim(0, 1)+ theme(text = element_text(size=22)) + 
  labs(x ="Verb Bias", y = "mBERT Probability (Spanish)") +
  scale_x_discrete(breaks=c("0","1"),
                   labels=c("Object-Bias", "Subject-Bias")) + 
  scale_fill_manual(values = c("#9999CC", "darkorchid3"), name= "Antecedent", labels=c("Object", "Subject"))

mbert_pronoun

mbert_pronoun <- ggplot(bert_it_score, aes(x=hasIC, y=score_mbert, fill=isHigh)) +
  geom_boxplot(notch=TRUE, outlier.size = 0.1) + ylim(0, 1)+ theme(text = element_text(size=22)) + 
  labs(x ="Verb Bias", y = "mBERT Probability (Italian)") +
  scale_x_discrete(breaks=c("0","1"),
                   labels=c("Object-Bias", "Subject-Bias")) + 
  scale_fill_manual(values = c("#9999CC", "darkorchid3"), name= "Antecedent", labels=c("Object", "Subject"))

mbert_pronoun
