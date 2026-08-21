"""Build final JUQ composite figures from archived summaries only."""
from __future__ import annotations
import math
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; TABLES=ROOT/"results"/"summary"; V2=TABLES/"numerical"; FP=TABLES/"first_principles"; OUT=ROOT/"figures"/"submission"; OUT.mkdir(parents=True,exist_ok=True); TIFF_OUT=os.environ.get("MECHAI_TIFF_OUT")
MM=1/25.4; WIDTH=183/25.4  # 183 mm, JUQ double-column width
INK="#252A34"; GRAY="#7A8088"; LIGHT="#D9DEE5"; BLUE="#275D8C"; BLUE2="#5B8DB8"; TEAL="#2F8F92"; PURPLE="#755C91"; RED="#B5514F"
CMAP=LinearSegmentedColormap.from_list("effective_blue",["#F5F7FA","#BCD3E5",BLUE])
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],"font.size":6.4,"axes.titlesize":6.9,"axes.labelsize":6.4,"xtick.labelsize":5.8,"ytick.labelsize":5.8,"legend.fontsize":5.8,"axes.linewidth":.65,"axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False,"svg.fonttype":"none","pdf.fonttype":42,"savefig.facecolor":"white"})
CRITERIA=["aic","aicc","bic","gic_pred","gic_evid","gic_eff_logn"]
CLABEL={"aic":"AIC","aicc":"AICc","bic":"BIC","gic_pred":"GIC-pred","gic_evid":"GIC-evid","gic_eff_logn":"Geometric BIC","gic_eff":"Legacy geometric BIC","gic_vol_050":"Volume sensitivity","ogic_e":"GIC-evid","waic_laplace":"Local WAIC"}
CCOLOR=dict(zip(CRITERIA,[LIGHT,"#B3B7BD",GRAY,BLUE,TEAL,PURPLE]))
SCENARIOS=["regular_sir_full","early_seir_infected_only","missing_time_varying_transmission","missing_neural_feedback","noisy_sir_overfit_risk"]
SLABEL={"regular_sir_full":"SIR\nfull","early_seir_infected_only":"SEIR\nearly observed","missing_time_varying_transmission":"Time-varying\ntransmission","missing_neural_feedback":"Neural\nfeedback","noisy_sir_overfit_risk":"SIR\nnoisy"}
STUDIES=["biochemical_haldane","biochemical_ude","ecology_rm","fhn_standard","fhn_ude"]
STLABEL={"biochemical_haldane":"Haldane","biochemical_ude":"Biochemical UDE","ecology_rm":"Predator-prey","fhn_standard":"FHN","fhn_ude":"FHN-UDE"}
PMETHODS=["equal","aic","bic","gic_evid","stacking","hard_gic_pred"]
PLABEL={"equal":"Equal","aic":"AIC","bic":"BIC","gic_evid":"GIC-evid","stacking":"Stacking","hard_gic_pred":"Hard GIC-pred"}
PCOLOR=dict(zip(PMETHODS,[LIGHT,"#B3B7BD",GRAY,TEAL,PURPLE,BLUE2])); MODEL_COLORS={"sir":LIGHT,"tv_sir":GRAY,"ude_sir_h2":BLUE,"neural_ode_h2":TEAL}; CANDIDATE_LABEL={"neural_ode_h2":"Neural ODE","seir":"SEIR","sir":"SIR","tv_sir":"Time-varying SIR","ude_sir_h2":"UDE-SIR","ude_appearance":"UDE appearance","minimal_gamma":"Minimal model, fitted","neural_ode":"Neural ODE","minimal_fixed":"Minimal model, fixed"}; DOMAIN_LABEL={"biochemical":"Biochemical","ecology":"Ecological","fhn":"Electrophysiological"}

def read(path,*required):
    d=pd.read_csv(path); missing=sorted(set(required)-set(d.columns))
    if missing: raise ValueError(f"{path.name}: missing {missing}")
    return d

def panel(ax,label): ax.text(-.11,1.07,label,transform=ax.transAxes,fontsize=8,fontweight="bold",ha="left",va="top",clip_on=False)
def clean(ax): ax.grid(axis="y",color="#E8EBEF",lw=.45,zorder=0); ax.set_axisbelow(True)
def heat(ax,matrix,vmin=0,vmax=1,annotate=False):
    im=ax.imshow(matrix.to_numpy(float),cmap=CMAP,vmin=vmin,vmax=vmax,aspect="auto")
    if annotate:
        for y in range(matrix.shape[0]):
            for x in range(matrix.shape[1]):
                v=matrix.iloc[y,x]
                if pd.notna(v): ax.text(x,y,f"{v:.2f}",ha="center",va="center",fontsize=5.5,color="white" if v>vmin+.62*(vmax-vmin) else INK)
    return im

def export(fig,stem):
    fig.savefig(OUT/f"{stem}.pdf",bbox_inches="tight",pad_inches=.025)
    fig.savefig(OUT/f"{stem}.svg",bbox_inches="tight",pad_inches=.025)
    fig.savefig(OUT/f"{stem}.png",dpi=600,bbox_inches="tight",pad_inches=.025)
    if TIFF_OUT:
        target=Path(TIFF_OUT); target.mkdir(parents=True,exist_ok=True)
        fig.savefig(target/f"{stem}.tiff",dpi=600,bbox_inches="tight",pad_inches=.025)
    plt.close(fig)
def entropy(x):
    p=x.value_counts(normalize=True).to_numpy(); return 0. if len(p)<2 else float(-(p*np.log(p)).sum()/np.log(len(p)))
def margins(frame,keys,score):
    def f(block):
        valid = block[score].notna(); v=np.sort(block.loc[valid, score]); return float(v[1]-v[0]) if len(v)>1 else np.nan
    return frame.groupby(keys,observed=True).apply(f,include_groups=False).rename("margin").reset_index()

def supplement1():
    cs=read(FP/"core_selection_summary.csv"); xs=read(FP/"biological_systems_selection_summary.csv"); core=read(FP/"core_scores.csv"); cross=read(FP/"biological_systems_scores.csv"); fig,ax=plt.subplots(2,2,figsize=(7.2047,5.7))
    for a,frame,key,rows,labels,title in [(ax[0,0],cs,"scenario",SCENARIOS,SLABEL,"Epidemic recovery"),(ax[0,1],xs,"study",STUDIES,STLABEL,"Recovery across biological systems")]:
        m=frame.pivot(index=key,columns="criterion",values="recovery_rate").reindex(rows)[CRITERIA]; im=heat(a,m,annotate=True); a.set_xticks(range(6),[CLABEL[z] for z in CRITERIA],rotation=35,ha="right"); a.set_yticks(range(len(rows)),[labels[z].replace("\n"," ") for z in rows]); a.set_title(title)
    for a,frame,key,rows,labels,title in [(ax[1,0],core,"scenario",SCENARIOS,SLABEL,"Core GIC-pred margins"),(ax[1,1],cross,"study",STUDIES,STLABEL,"Biological-system GIC-pred margins")]:
        m=margins(frame,[key,"seed"],"gic_pred"); a.boxplot([m.loc[m[key]==z,"margin"].clip(lower=1e-12) for z in rows],showfliers=False,patch_artist=True,boxprops={"facecolor":BLUE2},medianprops={"color":INK}); a.set_yscale("log"); a.set_xticks(range(1,len(rows)+1),[labels[z].replace("\n"," ") for z in rows],rotation=30,ha="right"); a.set(ylabel="Runner-up margin",title=title); clean(a)
    for i,a in enumerate(ax.flat): panel(a,chr(97+i))
    cax=fig.add_axes([.94,.58,.014,.31]); fig.colorbar(im,cax=cax,label="Recovery rate"); fig.subplots_adjust(left=.12,right=.91,bottom=.12,top=.95,wspace=.58,hspace=.56); export(fig,"figS1_recovery_margins")

def supplement2():
    d=read(FP/"phase_diagram_selection_summary.csv"); methods=["aic","bic","gic_pred","gic_evid"]
    fig,ax=plt.subplots(4,3,figsize=(7.2047,7.8),sharex=True,sharey=True)
    for r,meth in enumerate(methods):
        for c,t in enumerate([1,2,4]):
            m=d[(d.criterion==meth)&(d.trajectories==t)].pivot(index="noise",columns="n_times",values="recovery_rate").sort_index(ascending=False)
            im=heat(ax[r,c],m,annotate=True); ax[r,c].set_xticks(range(3),m.columns); ax[r,c].set_yticks(range(3),[f"{z:.3f}" for z in m.index])
            if r==0: ax[r,c].set_title(f"{t} independent {'trajectories' if t>1 else 'trajectory'}")
            if c==0: ax[r,c].set_ylabel(CLABEL[meth]+"\nNoise s.d.")
            if r==3: ax[r,c].set_xlabel("Observation times")
            panel(ax[r,c],chr(97+r*3+c))
    cax=fig.add_axes([.925,.12,.014,.76]); fig.colorbar(im,cax=cax,label="UDE recovery rate")
    fig.subplots_adjust(left=.105,right=.90,bottom=.07,top=.96,wspace=.16,hspace=.24); export(fig,"figS2_phase_all_criteria")
def supplement3():
    d=read(V2/"reference_sensitivity_summary.csv"); fig,ax=plt.subplots(2,4,figsize=(7.2047,4.8),sharex=True,sharey=True)
    for r,scenario in enumerate(["early_seir_infected_only","missing_neural_feedback"]):
        for c,gamma in enumerate([0,.25,.5,1]):
            m=d[(d.scenario==scenario)&np.isclose(d.gamma,gamma)].pivot(index="rho_nn",columns="resolution",values="mean").sort_index(ascending=False); im=heat(ax[r,c],m); ax[r,c].set_xticks(range(len(m.columns)),[f"{math.log10(z):.1f}" for z in m.columns],rotation=35,ha="right"); ax[r,c].set_yticks(range(len(m.index)),[f"{z:g}" for z in m.index]); ax[r,c].set_title(f"$\\gamma={gamma:g}$")
            if c==0: ax[r,c].set_ylabel(SLABEL[scenario].replace("\n"," ")+"\n$\\rho$")
            if r==1: ax[r,c].set_xlabel("$\\log_{10}\\lambda$")
            panel(ax[r,c],chr(97+r*4+c))
    cax=fig.add_axes([.93,.14,.014,.73]); fig.colorbar(im,cax=cax,label="Recovery rate"); fig.subplots_adjust(left=.12,right=.90,bottom=.12,top=.94,wspace=.18,hspace=.3); export(fig,"figS3_reference_full")

def supplement4():
    core=read(FP/"core_scores.csv"); cross=read(FP/"biological_systems_scores.csv"); fig,ax=plt.subplots(2,2,figsize=(7.2047,5.2)); specs=[(core,"candidate","gradient_norm","Core residuals"),(core,"candidate","wall_seconds","Core fit time"),(cross,"domain","gradient_norm","Biological-system residuals"),(cross,"domain","wall_seconds","Biological-system fit time")]
    for a,(frame,key,col,title) in zip(ax.flat,specs):
        groups=sorted(frame[key].unique()); a.boxplot([frame.loc[frame[key]==g,col].clip(lower=1e-12) for g in groups],showfliers=False,patch_artist=True,boxprops={"facecolor":BLUE2},medianprops={"color":INK}); a.set_yscale("log"); a.set_xticks(range(1,len(groups)+1),[CANDIDATE_LABEL.get(g,DOMAIN_LABEL.get(g,g.replace("_"," "))) for g in groups],rotation=30,ha="right"); a.set(ylabel="Gradient norm" if "residual" in title.lower() else "Wall time (s)",title=title); clean(a)
    for i,a in enumerate(ax.flat): panel(a,chr(97+i))
    fig.subplots_adjust(left=.1,right=.99,bottom=.14,top=.95,wspace=.4,hspace=.62); export(fig,"figS4_fit_diagnostics")

def supplement5():
    cap=read(TABLES/"capacity_study_summary.csv"); inv=read(TABLES/"invariance_redundancy_study.csv")
    fig,ax=plt.subplots(1,3,figsize=(7.2047,3.15),gridspec_kw={"width_ratios":[1,1,1.2]})
    x=np.arange(len(cap)); ax[0].bar(x-.18,cap.dimension,.36,color=LIGHT,label="Formal"); ax[0].bar(x+.18,cap.d_obs_mean,.36,yerr=cap.d_obs_std,color=BLUE,label="Effective",capsize=2)
    ax[0].set_xticks(x,[2,4,8]); ax[0].set(xlabel="Hidden width",ylabel="Dimension",title="Network capacity"); ax[0].legend(); clean(ax[0])
    coord=inv[inv.experiment=="coordinate_change"]; dims=sorted(coord.dimension.unique())
    ax[1].boxplot([coord.loc[coord.dimension==z,"relative_spectrum_error"].clip(lower=1e-16) for z in dims],showfliers=False,patch_artist=True,boxprops={"facecolor":TEAL},medianprops={"color":INK})
    ax[1].set_yscale("log"); ax[1].set_xticks(range(1,len(dims)+1),dims); ax[1].set(xlabel="Parameter dimension",ylabel="Relative spectrum error",title="Coordinate covariance"); clean(ax[1])
    red=inv[inv.experiment=="exact_duplication"].sort_values("dimension")
    ax[2].plot(red.dimension,red.raw_dimension_original,marker="o",color=GRAY,ls="--",label="Formal, original")
    ax[2].plot(red.dimension,red.raw_dimension_duplicate,marker="s",color=INK,label="Formal, duplicated")
    ax[2].plot(red.dimension,red.d_obs_original,marker="o",color=BLUE,ls="--",label="Effective, original")
    ax[2].plot(red.dimension,red.d_obs_duplicate,marker="s",color=TEAL,label="Effective, duplicated")
    ax[2].set(xlabel="Original parameter dimension",ylabel="Dimension after representation",title="Exact parameter duplication"); ax[2].legend(ncol=1,loc="upper left"); clean(ax[2])
    for i,a in enumerate(ax): panel(a,chr(97+i))
    fig.subplots_adjust(left=.08,right=.99,bottom=.16,top=.92,wspace=.48); export(fig,"figS5_geometry_checks")
def supplement6():
    intervention=read(TABLES/"intervention_selection_summary.csv"); metric=read(TABLES/"metric_boundary.csv"); glucose=read(TABLES/"glucose_case_summary.csv"); ecology=read(TABLES/"ecology_selection_summary.csv"); fig,ax=plt.subplots(2,2,figsize=(7.2047,5.2)); methods=[m for m in ["aic","aicc","bic","gic_eff","gic_vol_050","ogic_e"] if m in intervention.criterion.values]; d=intervention.set_index("criterion").reindex(methods); ax[0,0].bar(range(len(d)),d.recovery_rate,color=[LIGHT,"#B3B7BD",GRAY,BLUE,TEAL,PURPLE][:len(methods)]); ax[0,0].errorbar(range(len(d)),d.recovery_rate,yerr=[d.recovery_rate-d.recovery_ci_low,d.recovery_ci_high-d.recovery_rate],fmt="none",ecolor=INK,capsize=2); ax[0,0].set_xticks(range(len(d)),[{"aic":"AIC","aicc":"AICc","bic":"BIC","gic_eff":"Geometric BIC","gic_vol_050":"Volume sensitivity","ogic_e":"GIC-evid"}[m] for m in methods],rotation=30,ha="right"); ax[0,0].set(ylim=(0,1.05),ylabel="Recovery rate",title="Independent initial conditions"); clean(ax[0,0])
    x=np.arange(2); ax[0,1].bar(x-.18,metric.amplitude_information,.36,color=BLUE,label="Amplitude"); ax[0,1].bar(x+.18,metric.shift_information,.36,color=PURPLE,label="Shift"); ax[0,1].set_yscale("symlog",linthresh=1e-3); ax[0,1].set_xticks(x,[z.split()[0] for z in metric.metric]); ax[0,1].set(ylabel="Pullback information",title="Metric-dependent information"); ax[0,1].legend(); clean(ax[0,1])
    g=glucose.sort_values("blocked_mse"); ax[1,0].barh(range(len(g)),g.blocked_mse,xerr=g.blocked_mse_sd,color=[BLUE,TEAL,GRAY,LIGHT]); ax[1,0].set_yticks(range(len(g)),[CANDIDATE_LABEL.get(z,z.replace("_"," ")) for z in g.candidate]); ax[1,0].invert_yaxis(); ax[1,0].set(xlabel="Blocked prediction MSE",title="Glucose--insulin extrapolation"); clean(ax[1,0])
    legacy=["aic","aicc","bic","gic_eff","gic_vol_050","ogic_e"]; e=ecology[ecology.criterion.isin(legacy)].set_index("criterion").reindex(legacy); e=e[e["recovery_rate"].notna()]; ax[1,1].bar(range(len(e)),e.recovery_rate,color=[LIGHT,"#B3B7BD",GRAY,BLUE,TEAL,PURPLE][:len(e)]); ax[1,1].set_xticks(range(len(e)),[{"aic":"AIC","aicc":"AICc","bic":"BIC","gic_eff":"Geometric BIC","gic_vol_050":"Volume sensitivity","ogic_e":"GIC-evid"}[m] for m in e.index],rotation=30,ha="right"); ax[1,1].set(ylim=(0,1.05),ylabel="Recovery rate",title="Ecological model recovery"); clean(ax[1,1])
    for i,a in enumerate(ax.flat): panel(a,chr(97+i))
    fig.subplots_adjust(left=.11,right=.99,bottom=.15,top=.95,wspace=.52,hspace=.58); export(fig,"figS6_biological_boundaries")

def supplement7():
    cov=read(TABLES/"confidence_coverage_summary.csv"); s=read(ROOT/"results"/"summary"/"numerical"/"predictive_model_averaging_summary.csv"); fig,ax=plt.subplots(2,3,figsize=(7.2047,5.6),gridspec_kw={"width_ratios":[1,1,1.18]}); styles=[("naive_wald",GRAY,"s","Raw Wald"),("geometric_quotient",BLUE,"o","Quotient"),("simulation_calibrated",TEAL,"^","Calibrated")]
    for c,scenario in enumerate(["regular_sir_full","early_seir_infected_only"]):
        for method,color,marker,label in styles:
            d=cov[(cov.scenario==scenario)&(cov.method==method)].sort_values("nominal"); ax[0,c].plot(d.nominal,d.coverage,color=color,marker=marker,ms=3,lw=1,label=label); finite=np.isfinite(d.mean_log_relative_width); ax[1,c].plot(d.nominal[finite],d.mean_log_relative_width[finite],color=color,marker=marker,ms=3,lw=1,label=label)
        ax[0,c].plot([.5,.95],[.5,.95],ls="--",color=INK,lw=.7); ax[0,c].set(xlabel="Nominal",ylabel="Coverage",title=SLABEL[scenario].replace("\n"," ")); ax[1,c].set(xlabel="Nominal",ylabel="Mean log relative width",title="Region width"); clean(ax[0,c]); clean(ax[1,c])
    for r,metric in enumerate(["coverage90","width90"]):
        a=ax[r,2]; x=np.arange(len(PMETHODS)); w=.36
        for j,(scenario,color) in enumerate([("early_seir_infected_only",BLUE),("missing_neural_feedback",TEAL)]):
            d=s[s.scenario==scenario].set_index("method").reindex(PMETHODS); a.bar(x+(j-.5)*w,d[metric],w,color=color,label="Early SEIR" if j==0 else "Missing feedback")
        a.set_xticks(x,[PLABEL[m] for m in PMETHODS],rotation=35,ha="right"); a.set(ylabel="90% coverage" if metric=="coverage90" else "90% interval width",title="Predictive coverage" if r==0 else "Predictive sharpness"); a.tick_params(axis="x",pad=1); clean(a)
    ax[0,0].legend(); ax[0,2].set_ylim(0,1.22); ax[0,2].legend(ncol=2,loc="upper center")
    for i,a in enumerate(ax.flat): panel(a,chr(97+i))
    fig.subplots_adjust(left=.1,right=.99,bottom=.15,top=.94,wspace=.5,hspace=.56); export(fig,"figS7_uncertainty_full")

def supplement8():
    waic=read(V2/"waic_local_gaussian.csv"); sel=read(V2/"waic_local_gaussian_selections.csv"); scale=read(V2/"scalability_summary.csv"); timing=read(V2/"scalability_timings.csv"); fig,ax=plt.subplots(2,3,figsize=(7.2047,5.1)); recovery=sel.groupby("scenario").correct.mean().reindex(SCENARIOS); failure=(waic.status.ne("ok")|waic.waic_laplace.isna()).groupby(waic.scenario).mean().reindex(SCENARIOS)
    for a,values,title,color in [(ax[0,0],recovery,"Local-Gaussian WAIC recovery",BLUE),(ax[0,1],failure,"Nonfinite WAIC fraction",RED)]: a.barh(range(5),values,color=color); a.set_yticks(range(5),["SIR, full", "Early SEIR", "Time-varying", "Neural feedback", "SIR, noisy"]); a.invert_yaxis(); a.set(xlim=(0,1),title=title); clean(a)
    ax[0,2].plot(scale.dimension,scale.tensor_megabytes,marker="o",color=PURPLE); ax[0,2].set_yscale("log"); ax[0,2].set(xlabel="Parameter dimension",ylabel="Tensor memory (MB)",title="Memory scaling"); clean(ax[0,2])
    med=timing.groupby("dimension")[["jacobian_seconds","gram_seconds","spectrum_seconds"]].median()
    for col,color,label in [("jacobian_seconds",BLUE,"Jacobian"),("gram_seconds",TEAL,"Gram"),("spectrum_seconds",PURPLE,"Spectrum")]: ax[1,0].plot(med.index,med[col],marker="o",color=color,label=label)
    ax[1,0].set_yscale("log"); ax[1,0].set(xlabel="Parameter dimension",ylabel="Time (s)",title="Geometry computation"); ax[1,0].legend(); clean(ax[1,0])
    ax[1,1].plot(scale.dimension,scale.jacobian_median,marker="o",color=BLUE,label="Jacobian"); ax[1,1].plot(scale.dimension,scale.spectrum_median,marker="s",color=PURPLE,label="Spectrum"); ax[1,1].set_yscale("log"); ax[1,1].set(xlabel="Parameter dimension",ylabel="Median time (s)",title="Component scaling"); ax[1,1].legend(); clean(ax[1,1])
    counts=waic.groupby("scenario").size().reindex(SCENARIOS); ax[1,2].bar(range(5),counts,color=GRAY); ax[1,2].set_xticks(range(5),[SLABEL[z].replace("\n"," ") for z in SCENARIOS],rotation=38,ha="right"); ax[1,2].set(ylabel="Attempted fits",title="WAIC numerical sample"); clean(ax[1,2])
    for i,a in enumerate(ax.flat): panel(a,chr(97+i))
    fig.subplots_adjust(left=.11,right=.99,bottom=.17,top=.95,wspace=.52,hspace=.58); export(fig,"figS8_computation_waic")

def main():
    supplement1(); supplement2(); supplement3(); supplement4(); supplement5(); supplement6(); supplement7(); supplement8()
    print(f"Wrote 8 supplementary figures to {OUT}")
if __name__=="__main__": main()
