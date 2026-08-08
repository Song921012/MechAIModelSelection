"""Rebuild aggregates, figures, and integrity reports from archived records."""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def call(name:str,*args:str)->None: subprocess.run([sys.executable,str(ROOT/"analysis"/name),*args],cwd=ROOT,check=True)
def main()->None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--figures",action="store_true");p.add_argument("--tables",action="store_true");p.add_argument("--audit",action="store_true");a=p.parse_args(); all_steps=not(a.figures or a.tables or a.audit)
    if a.tables or all_steps: call("tables.py")
    if a.figures or all_steps: call("figures.py")
    if a.audit or all_steps: call("audit.py","--profile","submission")
if __name__=="__main__": main()
