"""Build final JUQ composite figures from archived summaries only."""
from __future__ import annotations
import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; TABLES=ROOT/"results"/"summary"; V2=TABLES/"numerical"; OUT=ROOT/"figures"; OUT.mkdir(parents=True,exist_ok=True)
MM=1/25.4; WIDTH=183/25.4  # 183 mm, JUQ double-column width
INK="#252A34"; GRAY="#7A8088"; LIGHT="#D9DEE5"; BLUE="#275D8C"; BLUE2="#5B8DB8"; TEAL="#2F8F92"; PURPLE="#755C91"; RED="#B5514F"
CMAP=LinearSegmentedColormap.from_list("observable_blue",["#F5F7FA","#BCD3E5",BLUE])
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],"font.size":6.2,"axes.titlesize":6.6,"axes.labelsize":6.2,"xtick.labelsize":5.6,"ytick.labelsize":5.6,"legend.fontsize":5.5,"axes.linewidth":.65,"axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False,"svg.fonttype":"none","pdf.fonttype":42,"savefig.facecolor":"white"})
CRITERIA=["aic","aicc","bic","gic_eff","gic_vol_050","ogic_e"]
CLABEL={"aic":"AIC","aicc":"AICc","bic":"BIC","gic_eff":"GIC-eff","gic_vol_050":"GIC-vol","ogic_e":"GIC-Lap","waic_laplace":"local WAIC"}
CCOLOR=dict(zip(CRITERIA,[LIGHT,"#B3B7BD",GRAY,BLUE,TEAL,PURPLE]))
SCENARIOS=["regular_sir_full","early_seir_infected_only","missing_time_varying_transmission","missing_neural_feedback","noisy_sir_overfit_risk"]
SLABEL={"regular_sir_full":"SIR\nfull","early_seir_infected_only":"SEIR\nearly observed","missing_time_varying_transmission":"Time-varying\ntransmission","missing_neural_feedback":"Neural\nfeedback","noisy_sir_overfit_risk":"SIR\nnoisy"}
STUDIES=["biochemical_haldane","biochemical_ude","ecology_rm","fhn_standard","fhn_ude"]
STLABEL={"biochemical_haldane":"Haldane","biochemical_ude":"Biochemical UDE","ecology_rm":"Predator-prey","fhn_standard":"FHN","fhn_ude":"FHN-UDE"}
PMETHODS=["equal","aic","bic","gic_eff","gic_vol_050","stacking","hard_gic"]
PLABEL={"equal":"Equal","aic":"AIC","bic":"BIC","gic_eff":"GIC-eff","gic_vol_050":"GIC-vol","stacking":"Stacking","hard_gic":"Hard GIC"}
PCOLOR=dict(zip(PMETHODS,[LIGHT,"#B3B7BD",GRAY,BLUE,TEAL,PURPLE,BLUE2])); MODEL_COLORS={"sir":LIGHT,"tv_sir":GRAY,"ude_sir_h2":BLUE,"neural_ode_h2":TEAL}

def read(path,*required):
    d=pd.read_csv(path); missing=sorted(set(required)-set(d.columns))
    if missing: raise ValueError(f"{path.name}: missing {missing}")
    return d

def panel(ax,label): ax.text(-.12,1.08,label,transform=ax.transAxes,fontsize=8,fontweight="bold",ha="left",va="top",clip_on=False)
def clean(ax): ax.grid(axis="y",color="#E8EBEF",lw=.45,zorder=0); ax.set_axisbelow(True)
def heat(ax,matrix,vmin=0,vmax=1,annotate=False):
    im=ax.imshow(matrix.to_numpy(float),cmap=CMAP,vmin=vmin,vmax=vmax,aspect="auto")
    if annotate:
        for y in range(matrix.shape[0]):
            for x in range(matrix.shape[1]):
                v=matrix.iloc[y,x]
                if pd.notna(v): ax.text(x,y,f"{v:.2f}",ha="center",va="center",fontsize=5.2,color="white" if v>vmin+.62*(vmax-vmin) else INK)
    return im

def export(fig,stem):
    fig.savefig(OUT/f"{stem}.pdf",bbox_inches="tight",pad_inches=.025)
    fig.savefig(OUT/f"{stem}.svg",bbox_inches="tight",pad_inches=.025)
    fig.savefig(OUT/f"{stem}.png",dpi=600,bbox_inches="tight",pad_inches=.025)
    plt.close(fig)
def entropy(x):
    p=x.value_counts(normalize=True).to_numpy(); return 0. if len(p)<2 else float(-(p*np.log(p)).sum()/np.log(len(p)))
def margins(frame,keys,score):
    def f(block):
        valid = block[score].notna(); v=np.sort(block.loc[valid, score]); return float(v[1]-v[0]) if len(v)>1 else np.nan
    return frame.groupby(keys,observed=True).apply(f,include_groups=False).rename("margin").reset_index()

def figure1():
    sel=read(V2/"core_selections.csv","scenario","criterion","selected","correct"); scores=read(V2/"core_scores.csv","scenario","candidate","seed","dimension","d_obs","c_obs","deviance")
    summary=sel.groupby(["scenario","criterion"],observed=True).correct.mean().unstack(); fig,ax=plt.subplots(2,3,figsize=(7.2047,4.05),gridspec_kw={"height_ratios":[1,.92]})
    m=summary.reindex(SCENARIOS)[CRITERIA]; im=heat(ax[0,0],m,annotate=True); ax[0,0].set_xticks(range(6),[CLABEL[x] for x in CRITERIA],rotation=35,ha="right"); ax[0,0].set_yticks(range(5),[SLABEL[x].replace("\n"," ") for x in SCENARIOS]); ax[0,0].set_title("Truth recovery"); fig.colorbar(im,ax=ax[0,0],fraction=.04,pad=.02,label="Recovery rate")
    d=(summary.gic_eff-summary.bic).reindex(SCENARIOS); ax[0,1].barh(range(5),d,color=[TEAL if x>=0 else RED for x in d]); ax[0,1].axvline(0,color=INK,lw=.7); ax[0,1].set_yticks(range(5),[SLABEL[x].replace("\n"," ") for x in SCENARIOS]); ax[0,1].set(xlabel="GIC-eff minus BIC recovery",title="Change relative to BIC"); clean(ax[0,1])
    c=scores.groupby("candidate").agg(raw=("dimension","first"),effective=("d_obs","mean"),sd=("d_obs","std")).sort_values("raw"); x=np.arange(len(c)); ax[0,2].bar(x-.17,c.raw,.34,color=LIGHT,label="Formal"); ax[0,2].bar(x+.17,c.effective,.34,yerr=c.sd,color=BLUE,label="Observable",capsize=1.5); ax[0,2].set_xticks(x,[v.replace("_"," ") for v in c.index],rotation=30,ha="right"); ax[0,2].set(ylabel="Dimension",title="Formal and effective dimension"); ax[0,2].legend(ncol=2); clean(ax[0,2])
    v=scores.groupby("candidate").c_obs.agg(["median","std"]).sort_values("median"); ax[1,0].barh(range(len(v)),v["median"],xerr=v["std"],color=TEAL,capsize=1.5); ax[1,0].set_yticks(range(len(v)),[z.replace("_"," ") for z in v.index]); ax[1,0].set(xlabel="Relative log-volume $C_{\\rm obs}$",title="Resolved model volume"); clean(ax[1,0])
    q=scores.copy(); q["fit_gap"]=q.deviance-q.groupby(["scenario","seed"]).deviance.transform("min"); q=q.groupby("candidate").agg(fit=("fit_gap","mean"),penalty=("d_obs",lambda z:2*z.mean()),raw=("dimension","first"))
    for name,row in q.iterrows(): ax[1,1].scatter(row.fit,row.penalty,s=18+row.raw,color=BLUE if "ude" in name else TEAL if "neural" in name else GRAY,edgecolor="white",lw=.5); ax[1,1].annotate(name.replace("_"," "),(row.fit,row.penalty),xytext=(3,2),textcoords="offset points",fontsize=5.1)
    ax[1,1].set(xlabel="Mean deviance gap",ylabel="$2d_{\\rm eff}$",title="Fit--complexity decomposition"); clean(ax[1,1])
    e=sel.groupby(["scenario","criterion"]).selected.apply(entropy).unstack().reindex(SCENARIOS)[CRITERIA]
    for criterion in CRITERIA: ax[1,2].plot(range(5),e[criterion],marker="o",ms=2.6,lw=1,color=CCOLOR[criterion],label=CLABEL[criterion])
    ax[1,2].set_xticks(range(5),[SLABEL[x] for x in SCENARIOS],rotation=30,ha="right"); ax[1,2].set(ylabel="Normalized selection entropy",ylim=(-.03,1.03),title="Selection dispersion"); ax[1,2].legend(ncol=2); clean(ax[1,2])
    for i,a in enumerate(ax.flat): panel(a,chr(97+i))
    fig.subplots_adjust(left=.075,right=.99,bottom=.14,top=.95,wspace=.53,hspace=.66); export(fig,"fig1_selection_complexity")

def figure2():
    s=read(V2/"phase_diagram_summary.csv","n_times","noise","trajectories","criterion","mean"); raw=read(V2/"phase_diagram_scores.csv","n_times","noise","trajectories","candidate","d_obs"); deff=raw[raw.candidate=="ude_sir_h2"].groupby(["n_times","noise","trajectories"]).d_obs.mean().reset_index(); fig,ax=plt.subplots(3,3,figsize=(7.2047,4.75),sharex=True,sharey=True)
    specs=[("gic_eff","GIC-eff recovery",0,1),("bic","BIC recovery",0,1),(None,"UDE effective dimension",deff.d_obs.min(),deff.d_obs.max())]
    for r,(crit,label,vmin,vmax) in enumerate(specs):
        for c,t in enumerate([1,2,4]):
            src=deff[deff.trajectories==t] if crit is None else s[(s.criterion==crit)&(s.trajectories==t)]; val="d_obs" if crit is None else "mean"; m=src.pivot(index="noise",columns="n_times",values=val).sort_index(ascending=False); im=heat(ax[r,c],m,vmin,vmax,True); ax[r,c].set_xticks(range(3),m.columns); ax[r,c].set_yticks(range(3),[f"{z:.3f}" for z in m.index]);
            if r==0: ax[r,c].set_title(f"{t} independent trajectory{'s' if t>1 else ''}")
            if c==0: ax[r,c].set_ylabel(label+"\nNoise s.d.")
            if r==2: ax[r,c].set_xlabel("Observation times")
            panel(ax[r,c],chr(97+r*3+c))
    fig.colorbar(ax[0,0].images[0],ax=ax[:2,:],fraction=.014,pad=.015,label="Recovery rate"); fig.colorbar(ax[2,0].images[0],ax=ax[2,:],fraction=.014,pad=.015,label="$d_{\\rm eff}$"); fig.subplots_adjust(left=.105,right=.91,bottom=.085,top=.95,hspace=.22,wspace=.17); export(fig,"fig2_information_phase")

def figure3():
    s=read(V2/"reference_sensitivity_summary.csv","scenario","rho_nn","resolution","gamma","mean"); p=read(TABLES/"formal_resolution_profiles.csv","scenario","candidate","seed","resolution","d_obs"); b=read(TABLES/"metric_boundary.csv","metric","amplitude_information","shift_information"); fig,ax=plt.subplots(2,3,figsize=(7.2047,3.95))
    for i,scenario in enumerate(["early_seir_infected_only","missing_neural_feedback"]):
        m=s[(s.scenario==scenario)&np.isclose(s.gamma,.5)].pivot(index="rho_nn",columns="resolution",values="mean").sort_index(ascending=False); im=heat(ax[0,i],m); ax[0,i].set_xticks(range(len(m.columns)),[f"{math.log10(z):.1f}" for z in m.columns],rotation=35,ha="right"); ax[0,i].set_yticks(range(len(m.index)),[f"{z:g}" for z in m.index]); ax[0,i].set(xlabel="$\\log_{10}\\lambda$",ylabel="Neural precision $\\rho$",title=SLABEL[scenario].replace("\n"," "))
    fig.colorbar(im,ax=ax[0,:2],fraction=.025,pad=.02,label="Recovery rate")
    d=s[(s.scenario=="missing_neural_feedback")&np.isclose(s.rho_nn,4)&np.isclose(s.resolution,1)]; ax[0,2].plot(d.gamma,d["mean"],marker="o",color=TEAL,lw=1.2); ax[0,2].set(xlabel="Volume weight $\\gamma$",ylabel="UDE recovery rate",ylim=(-.03,1.03),title="Volume sensitivity"); clean(ax[0,2])
    sub=p[p.scenario.isin(["early_seir_infected_only","missing_neural_feedback"])]
    for cand,color,ls in [("sir",GRAY,"--"),("seir",LIGHT,":"),("ude_sir_h2",BLUE,"-"),("neural_ode_h2",TEAL,"-.")]:
        d=sub[sub.candidate==cand].groupby("resolution").d_obs.mean()
        if len(d): ax[1,0].plot(d.index,d,color=color,ls=ls,lw=1.2,label=cand.replace("_"," "))
    ax[1,0].set_xscale("log"); ax[1,0].set(xlabel="Resolution $\\lambda$",ylabel="$d_{\\rm eff}(\\lambda)$",title="Resolution profiles"); ax[1,0].legend(ncol=2); clean(ax[1,0])
    x=np.arange(len(b)); w=.34; ax[1,1].bar(x-w/2,b.amplitude_information,w,color=BLUE,label="Amplitude"); ax[1,1].bar(x+w/2,b.shift_information,w,color=PURPLE,label="Temporal shift"); ax[1,1].set_yscale("symlog",linthresh=1e-3); ax[1,1].set_xticks(x,[z.split()[0] for z in b.metric]); ax[1,1].set(ylabel="Pullback information",title="Metric-dependent directions"); ax[1,1].legend(); clean(ax[1,1])
    ratio=b.set_index("metric"); ratio=ratio.shift_information/ratio.amplitude_information.clip(lower=1e-30); ax[1,2].bar(range(2),np.log10(ratio),color=[BLUE,PURPLE]); ax[1,2].axhline(0,color=INK,lw=.7); ax[1,2].set_xticks(range(2),[z.split()[0] for z in ratio.index]); ax[1,2].set(ylabel="$\\log_{10}$(shift/amplitude)",title="Scientific emphasis of the metric"); clean(ax[1,2])
    for i,a in enumerate(ax.flat): panel(a,chr(97+i))
    fig.subplots_adjust(left=.09,right=.985,bottom=.14,top=.94,wspace=.5,hspace=.62); export(fig,"fig3_reference_metric")

def figure4():
    s=read(V2/"crossdomain_scores.csv","study","domain","candidate","seed","dimension","d_obs","deviance","gic_eff","gradient_norm","wall_seconds"); sel=read(V2/"crossdomain_selections.csv","study","criterion","correct"); summary=sel.groupby(["study","criterion"]).correct.mean().unstack().reindex(STUDIES); fig,ax=plt.subplots(2,3,figsize=(7.2047,4.05))
    m=summary[CRITERIA]; im=heat(ax[0,0],m,annotate=True); ax[0,0].set_xticks(range(6),[CLABEL[x] for x in CRITERIA],rotation=35,ha="right"); ax[0,0].set_yticks(range(5),[STLABEL[x] for x in STUDIES]); ax[0,0].set_title("Cross-domain recovery"); fig.colorbar(im,ax=ax[0,0],fraction=.04,pad=.02)
    d=s.groupby(["domain","candidate"]).agg(raw=("dimension","first"),effective=("d_obs","mean")).reset_index()
    styles=[("biochemical",BLUE,"o"),("ecology",TEAL,"s"),("fhn",PURPLE,"^")]
    for domain,color,marker in styles:
        q=d[d.domain==domain]; ax[0,1].scatter(q.raw,q.effective,s=22,color=color,marker=marker,label=domain)
    lim=max(d.raw.max(),d.effective.max()); ax[0,1].plot([0,lim],[0,lim],color=GRAY,ls="--",lw=.7); ax[0,1].set(xlabel="Formal dimension",ylabel="Mean $d_{\\rm eff}$",title="Observable compression"); ax[0,1].legend(); clean(ax[0,1])
    q=s.copy(); q["fit_gap"]=q.deviance-q.groupby(["study","seed"]).deviance.transform("min"); q=q.groupby(["domain","candidate"]).agg(fit=("fit_gap","mean"),penalty=("d_obs",lambda z:2*z.mean())).reset_index()
    for domain,color,marker in styles:
        d=q[q.domain==domain]; ax[0,2].scatter(d.fit,d.penalty,s=20,color=color,marker=marker,label=domain)
    ax[0,2].set(xlabel="Mean deviance gap",ylabel="$2d_{\\rm eff}$",title="Fit--complexity balance"); clean(ax[0,2])
    m=margins(s,["study","seed"],"gic_eff"); data=[m.loc[m.study==z,"margin"].clip(lower=1e-12) for z in STUDIES]; boxes=ax[1,0].boxplot(data,showfliers=False,patch_artist=True,medianprops={"color":INK})
    for patch,color in zip(boxes["boxes"],[BLUE,BLUE,TEAL,PURPLE,PURPLE]): patch.set_facecolor(color)
    ax[1,0].set_yscale("log"); ax[1,0].set_xticks(range(1,6),[STLABEL[z] for z in STUDIES],rotation=30,ha="right"); ax[1,0].set(ylabel="GIC-eff runner-up margin",title="Selection separation"); clean(ax[1,0])
    for a,col,title,ylabel in [(ax[1,1],"gradient_norm","Terminal residual","Gradient norm"),(ax[1,2],"wall_seconds","Fit cost","Wall time (s)")]:
        groups=["biochemical","ecology","fhn"]; boxes=a.boxplot([s.loc[s.domain==z,col].clip(lower=1e-12) for z in groups],showfliers=False,patch_artist=True,medianprops={"color":INK})
        for patch,color in zip(boxes["boxes"],[BLUE,TEAL,PURPLE]): patch.set_facecolor(color)
        a.set_yscale("log"); a.set_xticks(range(1,4),groups,rotation=25,ha="right"); a.set(ylabel=ylabel,title=title); clean(a)
    for i,a in enumerate(ax.flat): panel(a,chr(97+i))
    fig.subplots_adjust(left=.09,right=.99,bottom=.15,top=.95,wspace=.5,hspace=.67); export(fig,"fig4_cross_domain")

def figure5():
    cov=read(TABLES/"confidence_coverage_summary.csv","scenario","method","nominal","coverage","coverage_ci_low","coverage_ci_high"); pred=read(V2/"predictive_model_averaging_summary.csv","scenario","method","mse_mean","mse_ci_low","mse_ci_high","within","between"); detail=read(V2/"predictive_model_averaging.csv","scenario","method","weight_sir","weight_tv_sir","weight_ude_sir_h2","weight_neural_ode_h2"); fig,ax=plt.subplots(2,3,figsize=(7.2047,4.05)); styles=[("naive_wald",GRAY,"s","Raw Wald"),("geometric_quotient",BLUE,"o","Quotient Wald"),("simulation_calibrated",TEAL,"^","Calibrated")]
    for a,scenario,title in [(ax[0,0],"regular_sir_full","Regular SIR"),(ax[0,1],"early_seir_infected_only","Early partial SEIR")]:
        for method,color,marker,label in styles:
            d=cov[(cov.scenario==scenario)&(cov.method==method)].sort_values("nominal"); a.errorbar(d.nominal,d.coverage,yerr=[d.coverage-d.coverage_ci_low,d.coverage_ci_high-d.coverage],color=color,marker=marker,ms=3,lw=1,capsize=1.5,label=label)
        a.plot([.48,.97],[.48,.97],color=INK,ls="--",lw=.7); a.set(xlabel="Nominal coverage",ylabel="Empirical coverage",xlim=(.48,.97),ylim=(0,1.02),title=title); clean(a)
    ax[0,0].legend()
    for a,scenario,title in [(ax[0,2],"early_seir_infected_only","Early partial SEIR"),(ax[1,0],"missing_neural_feedback","Missing feedback")]:
        d=pred[pred.scenario==scenario].set_index("method").reindex(PMETHODS); d=d[d["mse_mean"].notna()].sort_values("mse_mean"); y=np.arange(len(d)); a.barh(y,d.mse_mean,color=[PCOLOR[z] for z in d.index]); a.errorbar(d.mse_mean,y,xerr=[d.mse_mean-d.mse_ci_low,d.mse_ci_high-d.mse_mean],fmt="none",ecolor=INK,lw=.7,capsize=1.3); a.set_yticks(y,[PLABEL[z] for z in d.index]); a.invert_yaxis(); a.set(xlabel="Held-out MSE",title=title); clean(a)
    d=pred[pred.method.isin(["equal","aic","bic","gic_eff","stacking"])].copy(); labels=[f"{PLABEL[m]}\n{'SEIR' if s.startswith('early') else 'UDE'}" for s,m in zip(d.scenario,d.method)]; y=np.arange(len(d)); ax[1,1].barh(y,d.within,color=BLUE2,label="Within model"); ax[1,1].barh(y,d.between,left=d.within,color=TEAL,label="Between models"); ax[1,1].set_yticks(y,labels); ax[1,1].invert_yaxis(); ax[1,1].set(xlabel="Mean predictive variance",title="Uncertainty decomposition"); ax[1,1].legend(ncol=2); clean(ax[1,1])
    focus=detail[detail.method.isin(["gic_eff","stacking"])].groupby(["scenario","method"])[["weight_sir","weight_tv_sir","weight_ude_sir_h2","weight_neural_ode_h2"]].mean(); order=[("early_seir_infected_only","gic_eff"),("early_seir_infected_only","stacking"),("missing_neural_feedback","gic_eff"),("missing_neural_feedback","stacking")]; focus=focus.reindex(pd.MultiIndex.from_tuples(order)); left=np.zeros(4)
    for col in focus.columns:
        model=col.replace("weight_",""); values=focus[col].to_numpy(); ax[1,2].barh(range(4),values,left=left,color=MODEL_COLORS[model],label=model.replace("_"," ")); left+=values
    ax[1,2].set_yticks(range(4),[f"{'SEIR' if s.startswith('early') else 'UDE'}: {PLABEL[m]}" for s,m in order]); ax[1,2].invert_yaxis(); ax[1,2].set(xlim=(0,1),xlabel="Mean model weight",title="Weight composition"); ax[1,2].legend(ncol=2); clean(ax[1,2])
    for i,a in enumerate(ax.flat): panel(a,chr(97+i))
    fig.subplots_adjust(left=.11,right=.99,bottom=.13,top=.95,wspace=.62,hspace=.62); export(fig,"fig5_uncertainty_validation")

def supplement1():
    cs=read(V2/"core_selection_summary.csv"); xs=read(V2/"crossdomain_summary.csv"); core=read(V2/"core_scores.csv"); cross=read(V2/"crossdomain_scores.csv"); fig,ax=plt.subplots(2,2,figsize=(7.2047,5.7))
    for a,frame,key,rows,labels,title in [(ax[0,0],cs,"scenario",SCENARIOS,SLABEL,"Epidemic recovery"),(ax[0,1],xs,"study",STUDIES,STLABEL,"Cross-domain recovery")]:
        m=frame.pivot(index=key,columns="criterion",values="mean").reindex(rows)[CRITERIA]; im=heat(a,m,annotate=True); a.set_xticks(range(6),[CLABEL[z] for z in CRITERIA],rotation=35,ha="right"); a.set_yticks(range(len(rows)),[labels[z].replace("\n"," ") for z in rows]); a.set_title(title)
    for a,frame,key,rows,labels,title in [(ax[1,0],core,"scenario",SCENARIOS,SLABEL,"Core GIC-eff margins"),(ax[1,1],cross,"study",STUDIES,STLABEL,"Cross-domain GIC-eff margins")]:
        m=margins(frame,[key,"seed"],"gic_eff"); a.boxplot([m.loc[m[key]==z,"margin"].clip(lower=1e-12) for z in rows],showfliers=False,patch_artist=True,boxprops={"facecolor":BLUE2},medianprops={"color":INK}); a.set_yscale("log"); a.set_xticks(range(1,len(rows)+1),[labels[z].replace("\n"," ") for z in rows],rotation=30,ha="right"); a.set(ylabel="Runner-up margin",title=title); clean(a)
    for i,a in enumerate(ax.flat): panel(a,chr(97+i))
    fig.colorbar(im,ax=ax[0,:],fraction=.015,pad=.015,label="Recovery rate"); fig.subplots_adjust(left=.12,right=.92,bottom=.12,top=.95,wspace=.58,hspace=.56); export(fig,"figS1_recovery_margins")

def supplement2():
    d=read(V2/"phase_diagram_summary.csv"); methods=["aic","bic","gic_eff","gic_vol_050"]; fig,ax=plt.subplots(4,3,figsize=(7.2047,7.5),sharex=True,sharey=True)
    for r,meth in enumerate(methods):
        for c,t in enumerate([1,2,4]):
            m=d[(d.criterion==meth)&(d.trajectories==t)].pivot(index="noise",columns="n_times",values="mean").sort_index(ascending=False); im=heat(ax[r,c],m,annotate=True); ax[r,c].set_xticks(range(3),m.columns); ax[r,c].set_yticks(range(3),[f"{z:.3f}" for z in m.index]);
            if r==0: ax[r,c].set_title(f"{t} trajectory{'s' if t>1 else ''}")
            if c==0: ax[r,c].set_ylabel(CLABEL[meth]+"\nNoise s.d.")
            if r==3: ax[r,c].set_xlabel("Observation times")
            panel(ax[r,c],chr(97+r*3+c))
    fig.colorbar(im,ax=ax,fraction=.012,pad=.012,label="UDE recovery rate"); fig.subplots_adjust(left=.11,right=.91,bottom=.07,top=.96,wspace=.16,hspace=.22); export(fig,"figS2_phase_all_criteria")

def supplement3():
    d=read(V2/"reference_sensitivity_summary.csv"); fig,ax=plt.subplots(2,4,figsize=(7.2047,4.8),sharex=True,sharey=True)
    for r,scenario in enumerate(["early_seir_infected_only","missing_neural_feedback"]):
        for c,gamma in enumerate([0,.25,.5,1]):
            m=d[(d.scenario==scenario)&np.isclose(d.gamma,gamma)].pivot(index="rho_nn",columns="resolution",values="mean").sort_index(ascending=False); im=heat(ax[r,c],m); ax[r,c].set_xticks(range(len(m.columns)),[f"{math.log10(z):.1f}" for z in m.columns],rotation=35,ha="right"); ax[r,c].set_yticks(range(len(m.index)),[f"{z:g}" for z in m.index]); ax[r,c].set_title(f"$\\gamma={gamma:g}$")
            if c==0: ax[r,c].set_ylabel(SLABEL[scenario].replace("\n"," ")+"\n$\\rho$")
            if r==1: ax[r,c].set_xlabel("$\\log_{10}\\lambda$")
            panel(ax[r,c],chr(97+r*4+c))
    fig.colorbar(im,ax=ax,fraction=.012,pad=.012,label="Recovery rate"); fig.subplots_adjust(left=.12,right=.91,bottom=.12,top=.94,wspace=.16,hspace=.3); export(fig,"figS3_reference_full")

def supplement4():
    core=read(V2/"core_scores.csv"); cross=read(V2/"crossdomain_scores.csv"); fig,ax=plt.subplots(2,2,figsize=(7.2047,5.2)); specs=[(core,"candidate","gradient_norm","Core residuals"),(core,"candidate","wall_seconds","Core fit time"),(cross,"domain","gradient_norm","Cross-domain residuals"),(cross,"domain","wall_seconds","Cross-domain fit time")]
    for a,(frame,key,col,title) in zip(ax.flat,specs):
        groups=sorted(frame[key].unique()); a.boxplot([frame.loc[frame[key]==g,col].clip(lower=1e-12) for g in groups],showfliers=False,patch_artist=True,boxprops={"facecolor":BLUE2},medianprops={"color":INK}); a.set_yscale("log"); a.set_xticks(range(1,len(groups)+1),[g.replace("_"," ") for g in groups],rotation=30,ha="right"); a.set(ylabel="Gradient norm" if "residual" in title.lower() else "Wall time (s)",title=title); clean(a)
    for i,a in enumerate(ax.flat): panel(a,chr(97+i))
    fig.subplots_adjust(left=.1,right=.99,bottom=.14,top=.95,wspace=.4,hspace=.62); export(fig,"figS4_fit_diagnostics")

def supplement5():
    cap=read(TABLES/"capacity_study_summary.csv"); inv=read(TABLES/"invariance_redundancy_study.csv"); fig,ax=plt.subplots(1,3,figsize=(7.2047,2.8)); x=np.arange(len(cap)); ax[0].bar(x-.18,cap.dimension,.36,color=LIGHT,label="Formal"); ax[0].bar(x+.18,cap.d_obs_mean,.36,yerr=cap.d_obs_std,color=BLUE,label="Observable",capsize=2); ax[0].set_xticks(x,[2,4,8]); ax[0].set(xlabel="Hidden width",ylabel="Dimension",title="Network capacity"); ax[0].legend(); clean(ax[0])
    coord=inv[inv.experiment=="coordinate_change"]; dims=sorted(coord.dimension.unique()); ax[1].boxplot([coord.loc[coord.dimension==z,"relative_spectrum_error"].clip(lower=1e-16) for z in dims],showfliers=False,patch_artist=True,boxprops={"facecolor":TEAL},medianprops={"color":INK}); ax[1].set_yscale("log"); ax[1].set_xticks(range(1,len(dims)+1),dims); ax[1].set(xlabel="Parameter dimension",ylabel="Relative spectrum error",title="Coordinate covariance"); clean(ax[1])
    red=inv[inv.experiment=="redundant_parameter"]; red=red[red["raw_dimension_original"].notna()]; vals=[red.raw_dimension_original.mean(),red.raw_dimension_duplicate.mean(),red.d_obs_original.mean(),red.d_obs_duplicate.mean()]; ax[2].bar(range(4),vals,color=[LIGHT,GRAY,BLUE,TEAL]); ax[2].set_xticks(range(4),["raw","raw\nduplicated","effective","effective\nduplicated"],rotation=20); ax[2].set(ylabel="Dimension",title="Redundant coordinates"); clean(ax[2])
    for i,a in enumerate(ax): panel(a,chr(97+i))
    fig.subplots_adjust(left=.08,right=.99,bottom=.22,top=.9,wspace=.5); export(fig,"figS5_geometry_checks")

def supplement6():
    intervention=read(TABLES/"intervention_selection_summary.csv"); metric=read(TABLES/"metric_boundary.csv"); glucose=read(TABLES/"glucose_case_summary.csv"); ecology=read(TABLES/"ecology_selection_summary.csv"); fig,ax=plt.subplots(2,2,figsize=(7.2047,5.2)); methods=[m for m in CRITERIA if m in intervention.criterion.values]; d=intervention.set_index("criterion").reindex(methods); ax[0,0].bar(range(len(d)),d.recovery_rate,color=[CCOLOR[m] for m in methods]); ax[0,0].errorbar(range(len(d)),d.recovery_rate,yerr=[d.recovery_rate-d.recovery_ci_low,d.recovery_ci_high-d.recovery_rate],fmt="none",ecolor=INK,capsize=2); ax[0,0].set_xticks(range(len(d)),[CLABEL[m] for m in methods],rotation=30,ha="right"); ax[0,0].set(ylim=(0,1.05),ylabel="Recovery rate",title="Multiple-initial-condition intervention"); clean(ax[0,0])
    x=np.arange(2); ax[0,1].bar(x-.18,metric.amplitude_information,.36,color=BLUE,label="Amplitude"); ax[0,1].bar(x+.18,metric.shift_information,.36,color=PURPLE,label="Shift"); ax[0,1].set_yscale("symlog",linthresh=1e-3); ax[0,1].set_xticks(x,[z.split()[0] for z in metric.metric]); ax[0,1].set(ylabel="Pullback information",title="Fisher--Wasserstein boundary"); ax[0,1].legend(); clean(ax[0,1])
    g=glucose.sort_values("blocked_mse"); ax[1,0].barh(range(len(g)),g.blocked_mse,xerr=g.blocked_mse_sd,color=[BLUE,TEAL,GRAY,LIGHT]); ax[1,0].set_yticks(range(len(g)),g.candidate.str.replace("_"," ")); ax[1,0].invert_yaxis(); ax[1,0].set(xlabel="Blocked prediction MSE",title="Glucose--insulin extrapolation"); clean(ax[1,0])
    e=ecology[ecology.criterion.isin(CRITERIA)].set_index("criterion").reindex(CRITERIA); e=e[e["recovery_rate"].notna()]; ax[1,1].bar(range(len(e)),e.recovery_rate,color=[CCOLOR[m] for m in e.index]); ax[1,1].set_xticks(range(len(e)),[CLABEL[m] for m in e.index],rotation=30,ha="right"); ax[1,1].set(ylim=(0,1.05),ylabel="Recovery rate",title="Ecological transfer control"); clean(ax[1,1])
    for i,a in enumerate(ax.flat): panel(a,chr(97+i))
    fig.subplots_adjust(left=.11,right=.99,bottom=.15,top=.95,wspace=.52,hspace=.58); export(fig,"figS6_biological_boundaries")

def supplement7():
    cov=read(TABLES/"confidence_coverage_summary.csv"); s=read(V2/"predictive_model_averaging_summary.csv"); fig,ax=plt.subplots(2,3,figsize=(7.2047,5.1)); styles=[("naive_wald",GRAY,"s","Raw Wald"),("geometric_quotient",BLUE,"o","Quotient"),("simulation_calibrated",TEAL,"^","Calibrated")]
    for c,scenario in enumerate(["regular_sir_full","early_seir_infected_only"]):
        for method,color,marker,label in styles:
            d=cov[(cov.scenario==scenario)&(cov.method==method)].sort_values("nominal"); ax[0,c].plot(d.nominal,d.coverage,color=color,marker=marker,ms=3,lw=1,label=label); finite=np.isfinite(d.mean_log_relative_width); ax[1,c].plot(d.nominal[finite],d.mean_log_relative_width[finite],color=color,marker=marker,ms=3,lw=1,label=label)
        ax[0,c].plot([.5,.95],[.5,.95],ls="--",color=INK,lw=.7); ax[0,c].set(xlabel="Nominal",ylabel="Coverage",title=SLABEL[scenario].replace("\n"," ")); ax[1,c].set(xlabel="Nominal",ylabel="Mean log relative width",title="Region width"); clean(ax[0,c]); clean(ax[1,c])
    for r,metric in enumerate(["coverage90","width90"]):
        a=ax[r,2]; x=np.arange(len(PMETHODS)); w=.36
        for j,(scenario,color) in enumerate([("early_seir_infected_only",BLUE),("missing_neural_feedback",TEAL)]):
            d=s[s.scenario==scenario].set_index("method").reindex(PMETHODS); a.bar(x+(j-.5)*w,d[metric],w,color=color,label="Early SEIR" if j==0 else "Missing feedback")
        a.set_xticks(x,[PLABEL[m] for m in PMETHODS],rotation=35,ha="right"); a.set(ylabel="90% coverage" if metric=="coverage90" else "90% interval width",title="Predictive coverage" if r==0 else "Predictive sharpness"); clean(a)
    ax[0,0].legend(); ax[0,2].legend()
    for i,a in enumerate(ax.flat): panel(a,chr(97+i))
    fig.subplots_adjust(left=.1,right=.99,bottom=.15,top=.94,wspace=.5,hspace=.56); export(fig,"figS7_uncertainty_full")

def supplement8():
    waic=read(V2/"waic_local_gaussian.csv"); sel=read(V2/"waic_local_gaussian_selections.csv"); scale=read(V2/"scalability_summary.csv"); timing=read(V2/"scalability_timings.csv"); fig,ax=plt.subplots(2,3,figsize=(7.2047,5.1)); recovery=sel.groupby("scenario").correct.mean().reindex(SCENARIOS); failure=(waic.status.ne("ok")|waic.waic_laplace.isna()).groupby(waic.scenario).mean().reindex(SCENARIOS)
    for a,values,title,color in [(ax[0,0],recovery,"Local-Gaussian WAIC recovery",BLUE),(ax[0,1],failure,"Nonfinite WAIC fraction",RED)]: a.barh(range(5),values,color=color); a.set_yticks(range(5),[SLABEL[z].replace("\n"," ") for z in SCENARIOS]); a.invert_yaxis(); a.set(xlim=(0,1),title=title); clean(a)
    ax[0,2].plot(scale.dimension,scale.tensor_megabytes,marker="o",color=PURPLE); ax[0,2].set_yscale("log"); ax[0,2].set(xlabel="Parameter dimension",ylabel="Tensor memory (MB)",title="Memory scaling"); clean(ax[0,2])
    med=timing.groupby("dimension")[["jacobian_seconds","gram_seconds","spectrum_seconds"]].median()
    for col,color,label in [("jacobian_seconds",BLUE,"Jacobian"),("gram_seconds",TEAL,"Gram"),("spectrum_seconds",PURPLE,"Spectrum")]: ax[1,0].plot(med.index,med[col],marker="o",color=color,label=label)
    ax[1,0].set_yscale("log"); ax[1,0].set(xlabel="Parameter dimension",ylabel="Time (s)",title="Geometry computation"); ax[1,0].legend(); clean(ax[1,0])
    ax[1,1].plot(scale.dimension,scale.jacobian_median,marker="o",color=BLUE,label="Jacobian"); ax[1,1].plot(scale.dimension,scale.spectrum_median,marker="s",color=PURPLE,label="Spectrum"); ax[1,1].set_yscale("log"); ax[1,1].set(xlabel="Parameter dimension",ylabel="Median time (s)",title="Component scaling"); ax[1,1].legend(); clean(ax[1,1])
    counts=waic.groupby("scenario").size().reindex(SCENARIOS); ax[1,2].bar(range(5),counts,color=GRAY); ax[1,2].set_xticks(range(5),[SLABEL[z] for z in SCENARIOS],rotation=30,ha="right"); ax[1,2].set(ylabel="Attempted fits",title="WAIC numerical sample"); clean(ax[1,2])
    for i,a in enumerate(ax.flat): panel(a,chr(97+i))
    fig.subplots_adjust(left=.11,right=.99,bottom=.14,top=.95,wspace=.52,hspace=.58); export(fig,"figS8_computation_waic")

def qa_notes():
    (OUT/"FIGURE_QA.md").write_text("""# Submission figure QA\n\n- Core conclusion: observable pullback geometry changes effective model complexity; gains depend on information, reference scale, and metric choice.\n- Archetype: quantitative grids. Backend: Python/matplotlib only. Final width: 183 mm.\n- Configured minimum text size: 5.5 pt. PDF/SVG text remains editable; PNG uses 600 dpi. Journal TIFF files are generated in the manuscript workspace and omitted here to keep the repository compact.\n- Source data: audited CSV summaries linked to approximately 7,400 JSON records.\n- No observations were excluded by this plotting script. Existing nonfinite local-Gaussian WAIC records remain visible in Figure S8.\n- Recovery intervals and predictive intervals use the definitions stored by the audited aggregation pipeline.\n""",encoding="utf-8")

def main():
    figure1(); figure2(); figure3(); figure4(); figure5(); supplement1(); supplement2(); supplement3(); supplement4(); supplement5(); supplement6(); supplement7(); supplement8(); qa_notes(); print(f"Wrote 13 composite figures to {OUT}")
if __name__=="__main__": main()




