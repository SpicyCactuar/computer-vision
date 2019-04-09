#!/usr/bin/env python
#-*- coding: utf-8 -*

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import seaborn as sns


x = [3.5, 9.2, 11.7, 24.1, 49.1]
y = [u"Nuestra\nImplementación", u"HomographyNet\n(Regresión)" , u"ORB\n+\nRANSAC", u"HomographyNet\n(Clasificación)", u"Homografía\nIdentidad"]
ax = sns.barplot(y, x, palette="Blues")
ax.set(ylabel="Mean Average Corner Error\n(pixels)")
for i, rect in enumerate(ax.patches):
	txt = ax.text(rect.get_x() + rect.get_width()/2.0 - len(str(x[i])) * 0.05, rect.get_height()- 2.75 - rect.get_height()*0.05, x[i], fontsize=20, color='white')
	txt.set_path_effects([path_effects.Stroke(linewidth=2, foreground="gray"), path_effects.Normal()])

plt.xticks(rotation=30)
plt.tight_layout()
plt.show()
