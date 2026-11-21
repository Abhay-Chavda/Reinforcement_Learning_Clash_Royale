"""Turn raw results into the paper's tables and figures.

Inputs : results/evaluation/episodes.csv, results/train_monitor/*.monitor.csv
Outputs: results/evaluation/summary.csv, results/evaluation/summary.md,
         paper/tables/*.tex, paper/figures/*.{pdf,png}
Usage  : python -m experiments.analyze
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.evaluate import NAME_RE  # noqa: E402

FIG, TAB = Path("paper/figures"), Path("paper/tables")
CONDITIONS = [
    "PPO sparse / train vs random",
    "PPO shaped / train vs random",
    "PPO sparse / train vs ppo",
    "PPO shaped / train vs ppo",
]
SHORT = {
    "Random baseline": "Random",
    "Heuristic baseline": "Heuristic",
    "PPO sparse / train vs random": "Sparse | Rand",
    "PPO shaped / train vs random": "Shaped | Rand",
    "PPO sparse / train vs ppo": "Sparse | PPO",
    "PPO shaped / train vs ppo": "Shaped | PPO",
}
# Validated categorical slots 1-4 (light mode); baselines use neutral grays.
COLORS = {
    CONDITIONS[0]: "#2a78d6", CONDITIONS[1]: "#eb6834",
    CONDITIONS[2]: "#1baf7a", CONDITIONS[3]: "#eda100",
    "Random baseline": "#b4b3ad", "Heuristic baseline": "#6f6e69",
}
OPPONENTS = {"random": "Random opponent", "heuristic": "Heuristic opponent",
             "ppo_frozen": "Frozen-PPO opponent"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white", "legend.frameon": False,
})


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def summarise(df):
    df = df.assign(win=df.winner == "player", draw=df.winner == "draw",
                   loss=df.winner == "opponent",
                   hp_margin=df.player_tower_hp - df.opponent_tower_hp,
                   invalid_rate=df.invalid_actions / df.steps)
    per_seed = (df.groupby(["condition", "eval_opponent", "agent"])
                  .agg(win_rate=("win", "mean")).reset_index())
    rows = []
    for (cond, opp), g in df.groupby(["condition", "eval_opponent"]):
        seeds = per_seed[(per_seed.condition == cond) & (per_seed.eval_opponent == opp)].win_rate
        lo, hi = wilson(g.win.sum(), len(g))
        rows.append(dict(
            condition=cond, eval_opponent=opp, n_seeds=len(seeds), episodes=len(g),
            win_mean=seeds.mean(), win_std=seeds.std(ddof=1) if len(seeds) > 1 else 0.0,
            win_min=seeds.min(), win_max=seeds.max(), pooled_ci_low=lo, pooled_ci_high=hi,
            wins=int(g.win.sum()), draws=int(g.draw.sum()), losses=int(g.loss.sum()),
            king_kill_rate=g[g.win].king_destroyed.mean() if g.win.any() else 0.0,
            mean_hp_margin=g.hp_margin.mean(), mean_steps=g.steps.mean(),
            invalid_rate=g.invalid_rate.mean(),
        ))
    return pd.DataFrame(rows), per_seed, df


def order(conds):
    full = ["Random baseline", "Heuristic baseline"] + CONDITIONS
    return [c for c in full if c in set(conds)]


def write_tables(summary):
    TAB.mkdir(parents=True, exist_ok=True)
    conds = order(summary.condition)

    # Table: win rate (mean +- std over seeds) for every condition x opponent.
    lines = [r"\begin{tabular}{l" + "c" * len(OPPONENTS) + "}", r"\toprule",
             "Agent (reward $|$ training opponent) & " + " & ".join(
                 o.replace(" opponent", "") for o in OPPONENTS.values()) + r" \\", r"\midrule"]
    best = {o: summary[(summary.eval_opponent == o) &
                       summary.condition.isin(CONDITIONS)].win_mean.max() for o in OPPONENTS}
    for c in conds:
        cells = []
        for o in OPPONENTS:
            r = summary[(summary.condition == c) & (summary.eval_opponent == o)].iloc[0]
            txt = f"{100 * r.win_mean:.1f}"
            if r.n_seeds > 1:
                txt += rf" $\pm$ {100 * r.win_std:.1f}"
            if c in CONDITIONS and np.isclose(r.win_mean, best[o]):
                txt = r"\textbf{" + txt + "}"
            cells.append(txt)
        lines.append(SHORT[c].replace("|", r"$|$") + " & " + " & ".join(cells) + r" \\")
        if c == "Heuristic baseline":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "win_rates.tex").write_text("\n".join(lines) + "\n")

    # Table: behaviour / outcome detail against the random opponent.
    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r"Agent & W / D / L & 95\% CI (pooled) & King kills & HP margin & Invalid \\",
             r"\midrule"]
    for c in conds:
        r = summary[(summary.condition == c) & (summary.eval_opponent == "random")].iloc[0]
        lines.append(
            f"{SHORT[c].replace('|', '$|$')} & {r.wins} / {r.draws} / {r.losses} & "
            f"[{100 * r.pooled_ci_low:.1f}, {100 * r.pooled_ci_high:.1f}] & "
            f"{100 * r.king_kill_rate:.0f}\\% & {r.mean_hp_margin:+.0f} & "
            f"{100 * r.invalid_rate:.1f}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "detail_random.tex").write_text("\n".join(lines) + "\n")


def load_curves():
    curves = []
    for path in sorted(Path("results/train_monitor").glob("*.monitor.csv")):
        m = NAME_RE.match(path.name.replace(".monitor.csv", ""))
        if not m:
            continue
        mon = pd.read_csv(path, skiprows=1)
        mon["timestep"] = mon.l.cumsum()
        mon["win"] = (mon.winner == "player").astype(float)
        mon["rolling"] = mon.win.rolling(50, min_periods=10).mean()
        mon["condition"] = f"PPO {m['reward']} / train vs {m['train_opp']}"
        mon["seed"] = int(m["seed"])
        curves.append(mon)
    return pd.concat(curves) if curves else pd.DataFrame()


def fig_learning_curves(curves):
    grid = np.linspace(5_000, 100_000, 96)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    for ax, train_opp, title in zip(axes, ["random", "ppo"],
                                    ["Training vs random opponent", "Training vs frozen-PPO opponent"]):
        for c in [c for c in CONDITIONS if c.endswith(train_opp)]:
            sub = curves[curves.condition == c]
            ys = np.array([np.interp(grid, g.timestep, g.rolling.fillna(0.5))
                           for _, g in sub.groupby("seed")])
            mean, std = ys.mean(0), ys.std(0)
            ax.fill_between(grid, mean - std, mean + std, color=COLORS[c], alpha=0.18, lw=0)
            ax.plot(grid, mean, color=COLORS[c], lw=2,
                    label=("Shaped reward" if "shaped" in c else "Sparse reward"))
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("Environment steps")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("Training win rate\n(50-episode rolling)")
    fig.tight_layout()
    save(fig, "learning_curves")


def fig_win_rates(summary, per_seed):
    conds = order(summary.condition)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.9), sharey=True)
    x = np.arange(len(conds))
    for ax, (opp, title) in zip(axes, OPPONENTS.items()):
        for i, c in enumerate(conds):
            r = summary[(summary.condition == c) & (summary.eval_opponent == opp)].iloc[0]
            ax.bar(i, r.win_mean, width=0.68, color=COLORS[c], edgecolor="white", linewidth=2)
            seeds = per_seed[(per_seed.condition == c) & (per_seed.eval_opponent == opp)].win_rate
            if len(seeds) > 1:
                jitter = np.linspace(-0.18, 0.18, len(seeds))
                ax.scatter(i + jitter, seeds, s=10, color=INK, zorder=3, linewidths=0)
            ax.text(i, min(r.win_mean + 0.03, 1.02), f"{100 * r.win_mean:.0f}",
                    ha="center", va="bottom", fontsize=7, color=MUTED)
        ax.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (3, 3)))
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_xticks(x, [SHORT[c] for c in conds], rotation=50, ha="right", fontsize=7.5)
        ax.set_ylim(0, 1.1)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Evaluation win rate")
    fig.tight_layout()
    save(fig, "eval_win_rates")


def fig_actions(df):
    conds = order(df.condition)
    sub = df[df.eval_opponent == "random"]
    parts = {"Wait": sub.act_wait,
             "Knight": sub.act_knight_left + sub.act_knight_right,
             "Giant": sub.act_giant_left + sub.act_giant_right}
    tot = sub.steps
    shares = pd.DataFrame({k: v / tot for k, v in parts.items()}).assign(condition=sub.condition)
    shares = shares.groupby("condition").mean().loc[conds]
    # Deployments only (exclude wait) for a readable second panel.
    deploy = pd.DataFrame({k: v for k, v in parts.items() if k != "Wait"}).assign(condition=sub.condition)
    deploy = deploy.groupby("condition").mean().loc[conds]
    colors = {"Wait": "#c9c8c2", "Knight": "#6250d6", "Giant": "#e87ba4"}

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    y = np.arange(len(conds))
    left = np.zeros(len(conds))
    for k in ["Wait", "Knight", "Giant"]:
        axes[0].barh(y, shares[k], left=left, color=colors[k], edgecolor="white", linewidth=1.5, label=k)
        left += shares[k].values
    axes[0].set_yticks(y, [SHORT[c] for c in conds], fontsize=7.5)
    axes[0].invert_yaxis()
    axes[0].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].set_title("Share of decisions (vs random)", fontsize=9, color=INK, loc="left")
    axes[0].legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=8)
    axes[0].grid(axis="y", visible=False)

    w = 0.38
    axes[1].barh(y - w / 2, deploy.Knight, height=w, color=colors["Knight"], label="Knight")
    axes[1].barh(y + w / 2, deploy.Giant, height=w, color=colors["Giant"], label="Giant")
    axes[1].set_yticks(y, [])
    axes[1].invert_yaxis()
    axes[1].set_title("Attempted deployments per game", fontsize=9, color=INK, loc="left")
    axes[1].legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=8)
    axes[1].grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "action_distribution")


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    episodes = pd.read_csv("results/evaluation/episodes.csv")
    summary, per_seed, df = summarise(episodes)
    summary["order"] = summary.condition.map({c: i for i, c in enumerate(order(summary.condition))})
    summary = summary.sort_values(["order", "eval_opponent"]).drop(columns="order")
    summary.to_csv("results/evaluation/summary.csv", index=False)
    per_seed.to_csv("results/evaluation/per_seed_win_rates.csv", index=False)

    md = ["| Agent | " + " | ".join(OPPONENTS.values()) + " |", "|---" * (len(OPPONENTS) + 1) + "|"]
    for c in order(summary.condition):
        cells = []
        for o in OPPONENTS:
            r = summary[(summary.condition == c) & (summary.eval_opponent == o)].iloc[0]
            cells.append(f"{100 * r.win_mean:.1f}%" + (f" ± {100 * r.win_std:.1f}" if r.n_seeds > 1 else ""))
        md.append(f"| {c} | " + " | ".join(cells) + " |")
    Path("results/evaluation/summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    write_tables(summary)
    fig_win_rates(summary, per_seed)
    fig_actions(df)
    curves = load_curves()
    if not curves.empty:
        fig_learning_curves(curves)


if __name__ == "__main__":
    main()
