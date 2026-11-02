import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

BASE = os.path.dirname(os.path.abspath(__file__))
panel = pd.read_csv(os.path.join(BASE, 'entry_hazard_panel_v2_CORRECTED.csv'))
second_sample = panel[panel['n_timed_entrants'] >= 1].copy()

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, (dur, ev, title) in zip(axes, [
        ('first_entry_time', 'first_entry_event', 'Time to FIRST dispensary\n(days since state retail launch)'),
        ('second_gap_time', 'second_gap_event', 'Time from 1st to 2nd dispensary\n(days, among counties with a 1st entrant)')]):
    sample = panel if dur == 'first_entry_time' else second_sample
    for grp, label, color in [(1, 'border (<=70mi)', 'tab:red'), (0, 'interior', 'tab:blue')]:
        kmf = KaplanMeierFitter()
        mask = sample['border_70mi'] == grp
        kmf.fit(sample.loc[mask, dur], sample.loc[mask, ev], label=label)
        kmf.plot_survival_function(ax=ax, color=color)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Days')
    ax.set_ylabel('Share not yet entered')
fig.suptitle('Entry timing, border vs. interior counties (Kaplan-Meier)\n'
             'Corrected: excludes CO/WA records with placeholder (non-observed) open dates')
fig.tight_layout()
fig.savefig(os.path.join(BASE, 'fig_entry_hazard_km_CORRECTED.png'), dpi=150)
print("Saved fig_entry_hazard_km_CORRECTED.png")
