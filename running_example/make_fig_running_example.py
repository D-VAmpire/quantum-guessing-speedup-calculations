r"""make_fig_running_example.py -- the running-example figure (Fig. 1 of the arXiv full version, v2 on).

Writes fig_running_example.tex (standalone TikZ) into the directory given as argument, by default
next to this script. Compile it with pdflatex (needs the standalone, tikz, libertine, newtx and zi4
packages) to get fig_running_example.pdf, the file the paper includes. The PQQS '26 version has no
running example; toy_numbers.py checks the numbers the text states about it.

Toy advice, surprisals in units of h = ln 2:
  p1: one symbol at 1, three at 3, two at 4      -> (1/2, 1/8, 1/8, 1/8, 1/16, 1/16)
  p2: one symbol each at 1..5, two at 6          -> (1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/64)
N = 42 keys on 9 levels t/h = 2..10 with counts 1,1,4,6,6,7,5,8,4.

Layout: top row the input (each symbol one bar of height p_i(a) over its surprisal), middle rows the
count and mass laws of each coordinate and their convolution, bottom row the 42 keys in likelihood
order (rank ticks = N F_S(t)); the level t = 5h is highlighted in the S column and in the bottom row,
with its rank block J_t.

Usage:  python3 running_example/make_fig_running_example.py [output directory]
"""
import os
import sys
from fractions import Fraction as Fr
from itertools import product

p1 = [1, 3, 3, 3, 4, 4]          # surprisal of each symbol in units of h = ln 2, coordinate 1
p2 = [1, 2, 3, 4, 5, 6, 6]       # coordinate 2
HL = 5                            # highlighted level t = 5h
def count_law(bits, lo, hi):
    return [sum(1 for b in bits if b == k) for k in range(lo, hi + 1)]
c1, c2 = count_law(p1, 1, 4), count_law(p2, 1, 6)
cS = count_law([a + b for a, b in product(p1, p2)], 2, 10)
m1 = [Fr(c, 2**k) for c, k in zip(c1, range(1, 5))]
m2 = [Fr(c, 2**k) for c, k in zip(c2, range(1, 7))]
mS = [Fr(c, 2**k) for c, k in zip(cS, range(2, 11))]
assert sum(m1) == sum(m2) == sum(mS) == 1 and sum(cS) == 42

D = 0.33
HIN, HA = 0.42, 0.46
YA, YC, YM = 1.98, 0.98, 0.0
X1 = 0.80
X2 = X1 + 4*D + 0.55
X3 = X2 + 6*D + 0.55
XR = X3 + 9*D                    # right edge of the plots
out = []; w = out.append
def axis(x0, y0, n):
    w(rf"\draw[axisc] ({x0:.3f},{y0:.3f}) -- ({x0 + n*D:.3f},{y0:.3f});")
def hist(x0, y0, vals, color, nums=None, lo=None, hl_color=None):
    vmax = max(vals)
    for k, v in enumerate(vals):
        h = float(v / vmax) * HA
        col = hl_color if (hl_color and lo + k == HL) else color
        if h > 0:
            w(rf"\fill[{col}] ({x0 + k*D + 0.035:.3f},{y0:.3f}) rectangle ({x0 + (k+1)*D - 0.035:.3f},{y0 + h:.3f});")
        if nums is not None:
            w(rf"\node[lab6,anchor=south] at ({x0 + (k+0.5)*D:.3f},{y0 + h:.3f}) {{{nums[k]}}};")
    axis(x0, y0, len(vals))
def symbols(x0, y0, bits, lo, hi):
    for k in range(lo, hi + 1):
        n = sum(1 for b in bits if b == k)
        if n == 0:
            continue
        bw = (D - 0.07) / n
        for j in range(n):
            xa = x0 + (k - lo)*D + 0.035 + j*bw
            h = max(HIN * 2.0 ** (1 - k), 0.012)
            w(rf"\fill[accent!85!black] ({xa + 0.008:.3f},{y0:.3f}) rectangle ({xa + bw - 0.008:.3f},{y0 + h:.3f});")
    axis(x0, y0, hi - lo + 1)
def ticks(x0, y0, lo, n):
    for k in range(n):
        w(rf"\node[lab6,anchor=north] at ({x0 + (k+0.5)*D:.3f},{y0 - 0.02:.3f}) {{{lo + k}}};")
def divider(y):
    w(rf"\draw[axisc!80,line width=0.4pt,dash pattern=on 2pt off 1.6pt] (0.00,{y:.3f}) -- ({XR:.3f},{y:.3f});")

# ---------------- input: each symbol one bar of height p_i(a), over its surprisal
w(rf"\node[lab7,anchor=east] at ({X1 - 0.06:.2f},{YA + 0.18:.2f}) {{$p_i(a)$}};")
symbols(X1, YA, p1, 1, 4)
symbols(X2, YA, p2, 1, 6)
w(rf"\node[lab7,anchor=north] at ({X1 + 2*D:.3f},{YA - 0.02:.3f}) {{$p_1$}};")
w(rf"\node[lab7,anchor=north] at ({X2 + 3*D:.3f},{YA - 0.02:.3f}) {{$p_2$}};")
divider(YA - 0.36)

# ---------------- count and mass laws over surprisal, and their convolution
w(rf"\node[lab7,anchor=east] at ({X1 - 0.06:.2f},{YC + 0.18:.2f}) {{count}};")
w(rf"\node[lab7,anchor=east] at ({X1 - 0.06:.2f},{YM + 0.18:.2f}) {{mass}};")
hist(X1, YC, c1, "accent!50", c1); hist(X1, YM, m1, "accent!85")
hist(X2, YC, c2, "accent!50", c2); hist(X2, YM, m2, "accent!85")
hist(X3, YC, cS, "accent!50", cS, lo=2, hl_color="emer!55"); hist(X3, YM, mS, "accent!85", lo=2, hl_color="emer!85")
for x0, n, lo in ((X1, 4, 1), (X2, 6, 1), (X3, 9, 2)):
    ticks(x0, YM, lo, n)
w(rf"\node[sym] at ({(X1 + 4*D + X2)/2:.3f},{(YC + YM + HA)/2:.3f}) {{$*$}};")
w(rf"\node[sym] at ({(X2 + 6*D + X3)/2:.3f},{(YC + YM + HA)/2:.3f}) {{$=$}};")
w(rf"\node[lab7,anchor=north] at ({X1 + 2*D:.3f},-0.22) {{$s_1/h$}};")
w(rf"\node[lab7,anchor=north] at ({X2 + 3*D:.3f},-0.22) {{$s_2/h$}};")
w(rf"\node[lab7,anchor=north] at ({X3 + 4.5*D:.3f},-0.22) {{$S/h$}};")
divider(-0.62)

# ---------------- the 42 keys in likelihood order
YB = -2.02
HB = 0.95
XB0, XB1 = X1, XR
keys = sorted(a + b for a, b in product(p1, p2))
nb, ng, gap_frac = 42, 8, 0.55
pitch = (XB1 - XB0) / (nb + gap_frac * ng)
xs = []; x = XB0; prev = keys[0]
for t in keys:
    if t != prev:
        x += gap_frac * pitch; prev = t
    xs.append(x); x += pitch
lev = {}
for i, t in enumerate(keys):
    lev.setdefault(t, []).append(i)
for i, t in enumerate(keys):
    h = HB * 2.0 ** (2 - t)
    col = "emer!85" if t == HL else "accent!70"
    w(rf"\fill[{col}] ({xs[i] + 0.1*pitch:.3f},{YB:.3f}) rectangle ({xs[i] + 0.9*pitch:.3f},{YB + max(h, 0.014):.3f});")
w(rf"\draw[axisc] ({XB0:.3f},{YB:.3f}) -- ({XB1:.3f},{YB:.3f});")
w(rf"\node[lab6,anchor=east] at ({XB0 - 0.02:.3f},{YB + HB + 0.20:.3f}) {{$t/h$}};")
r = 0
for t in sorted(lev):
    idx = lev[t]; xa, xb = xs[idx[0]], xs[idx[-1]] + pitch
    w(rf"\node[lab6] at ({(xa + xb)/2:.3f},{YB + HB + 0.20:.3f}) {{{t}}};")
    if len(idx) > 1:
        w(rf"\draw[axisc!70,line width=0.3pt] ({xa + 0.02:.3f},{YB + HB + 0.10:.3f}) -- ({xb - 0.02:.3f},{YB + HB + 0.10:.3f});")
    r += len(idx)
    xr = xs[idx[-1]] + 0.5 * pitch
    w(rf"\draw[axisc] ({xr:.3f},{YB:.3f}) -- ({xr:.3f},{YB - 0.05:.3f});")
    w(rf"\node[lab6,anchor=north] at ({xr:.3f},{YB - 0.05:.3f}) {{{r}}};")
w(rf"\node[lab7,anchor=north] at ({(XB0 + XB1)/2:.3f},{YB - 0.24:.3f}) {{rank $r$}};")
w(rf"\node[lab7,anchor=east] at ({X1 - 0.06:.2f},{YB + 0.40:.2f}) {{$P(x)$}};")
idx = lev[HL]; xa, xb = xs[idx[0]], xs[idx[-1]] + pitch
hb = HB * 2.0 ** (2 - HL)
w(rf"\draw[emer!70!black,line width=0.45pt,decorate,decoration={{brace,amplitude=2pt}}] ({xa + 0.01:.3f},{YB + hb + 0.04:.3f}) -- ({xb - 0.01:.3f},{YB + hb + 0.04:.3f});")
w(rf"\node[lab7,anchor=south,text=emer!55!black] at ({(xa + xb)/2:.3f},{YB + hb + 0.09:.3f}) {{$J_t$}};")

body = "\n".join(out)
doc = r"""\documentclass[border=1pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage[tt=false, type1=true]{libertine}
\usepackage[varqu]{zi4}
\usepackage[libertine]{newtxmath}
\usepackage{tikz}
\usetikzlibrary{decorations.pathreplacing}
\definecolor{accent}{HTML}{5A5FC7}
\definecolor{emer}{HTML}{00A388}
\definecolor{axisc}{HTML}{95A5A6}
\definecolor{gtext}{HTML}{4A4A4A}
\tikzset{lab7/.style={font=\fontsize{7}{8}\selectfont,text=gtext,inner sep=1pt},
         lab6/.style={font=\fontsize{5.6}{6}\selectfont,text=gtext,inner sep=1pt},
         sym/.style={font=\fontsize{9}{10}\selectfont,text=gtext},
         axisc/.style={draw=axisc,line width=0.5pt}}
\begin{document}
\begin{tikzpicture}[x=1cm,y=1cm]
""" + body + r"""
\end{tikzpicture}
\end{document}
"""
outdir = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
path = os.path.join(outdir, "fig_running_example.tex")
open(path, "w").write(doc)
print(f"written {path}")
