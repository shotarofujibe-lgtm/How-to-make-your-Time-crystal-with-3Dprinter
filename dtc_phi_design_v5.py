#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
φ model 時間結晶シミュレータ v5 — 「周期の倍率 / 持続時間 / 持続周期」のどれか1つで回る

支配方程式（無次元, 1D, 周期境界。Veenstra, Binysh, ..., Coulais, arXiv:2508.20052 SM 式 S20）:
    ∇²[(1 - iα)φ + φ̇ + |φ|²φ] = φ̈ + Γ φ̇

■ 入力（どれか1つ以上。残りは自動で決まり、結果として表示される）
    --ratio    m   応答周期は入力周期の何倍か          m = T_resp / T_in
    --duration D   持続時間（無次元時間）              時間結晶状態を保つ時間 = シミュレーション時間
    --cycles   P   持続周期（応答の何周期ぶん持続か）  P = D / T_resp

  用語:
    入力周期 T_in  … 基準となる入力（駆動）周期。φ モデルは自励系で外部駆動を持たないため、
                     無次元時間の基準として与える（既定 T_in = 1。--Tin で変更）。
    応答周期 T_resp = 2π/ω   … 時間結晶状態の振動周期。
  関係:  T_resp = m·T_in = D/P
    ・m を指定        → ω = 2π/(m·T_in) が決まる
    ・D と P を指定   → ω = 2πP/D が決まる
    ・D だけ / P だけ → ω は既定（α=1, Γ=0 → ω=1）。D↔P は T_resp で換算
    ・3つとも指定する場合は互いに矛盾しないこと

■ 結果として表示されるもの（入力ではない）
    α, Γ, L, n, q, ω, A（厳密関係式 S24 で結ばれる設計点）と、m, D, P, T_resp、
    二次安定性、dt 固定での格子(dx, N)、計算時間の見積もり、完成可能性、実測の持続時間。

■ 時間刻みは dt = 0.01 固定（RK4）
    dt が固定なので、CFL 条件を満たすよう空間刻み dx 側を自動で調整する。
    満たせない（格子が粗すぎる / 1周期が数ステップしかない 等）場合は「完成可能性: 極めて低い」と表示する。

■ 完成可能性の事前判定（実行前に表示）
    存在 / dt固定での数値安定性・精度 / 持続（二次安定性） / 計算時間（実測ベンチから見積もり）
    の4観点を 高い・やや注意・低い・極めて低い で評価。計算不能・数値破綻が確実なものは、
    既定では実行せず中止する（--force で強制実行）。

■ 残り時間
    実行中は 進捗%・経過・残り時間・終了予定時刻 を常時表示（ベンチ値で初期化し、実測速度で更新）。

使い方:
  python dtc_phi_design_v5.py --ratio 2                  # 応答周期が入力周期の2倍
  python dtc_phi_design_v5.py --duration 1000            # 持続時間だけ決める
  python dtc_phi_design_v5.py --cycles 50                # 持続周期だけ決める
  python dtc_phi_design_v5.py --duration 600 --cycles 100  # 応答周期 = 6 が決まる
  python dtc_phi_design_v5.py                            # 引数なし → 対話入力（空欄=自動）
"""
import argparse
import math
import sys
import time
from collections import deque

import numpy as np

TWO_PI = 2.0 * math.pi
REL_TOL = 1e-6
DT = 0.01          # 時間刻み（固定）
CFL = 1.5          # stable_dt と同じ CFL 係数

LEVELS = ["高い", "やや注意", "低い", "極めて低い"]


# ------------------------------------------------------------------
#  厳密関係式と線形理論
# ------------------------------------------------------------------
def q_discrete(q, dx):
    """3点差分ラプラシアンの固有値に対応する有効波数。"""
    return math.sqrt(2.0 - 2.0 * math.cos(q * dx)) / dx


def travelling_wave(q, alpha, Gamma):
    """厳密進行波 (S24)。(ω, A)。A が実数にならなければ A=None。"""
    w = alpha * q**2 / (Gamma + q**2)
    a2 = (w / q) ** 2 - 1.0
    return w, (math.sqrt(a2) if a2 > 0 else None)


def sigma_plus(q, alpha, Gamma):
    q = np.asarray(q, dtype=float)
    root = np.sqrt((-4 * q**2 + q**4 + 4j * q**2 * alpha + 2 * q**2 * Gamma + Gamma**2).astype(complex))
    return 0.5 * (-q**2 - Gamma + root)


def band(alpha, Gamma):
    """線形不安定帯 (q-, q+)。不安定でなければ None。"""
    if alpha <= 2.0 * math.sqrt(Gamma):
        return None
    d = math.sqrt(alpha**2 - 4.0 * Gamma)
    return 0.5 * (alpha - d), 0.5 * (alpha + d)


def secondary_growth(q, alpha, Gamma, L, kmax=4.0):
    """
    進行波 (q) の二次（Benjamin-Feir 型）安定性。SM 式 S32 の 2次固有値問題を、
    周期箱 L で許される摂動波数 k=2πj/L (j≥1) について解き、最大成長率 max Re σ を返す。
    戻り値 (λ_max, そのときの k)。λ_max>0 なら不安定（e 倍時間 1/λ）。進行波が存在しなければ (None, None)。
    ※Γ=0, n=3 (L=200) で +0.0039 → 直接シミュレーションの成長率と一致を確認済み。
    ※L が非常に大きいときは j を間引いて評価する（小さい k を優先して全部調べる）。
    """
    w = alpha * q * q / (Gamma + q * q)
    A2 = (w / q) ** 2 - 1.0
    if A2 <= 0:
        return None, None
    jmax = int(kmax * L / TWO_PI)
    if jmax > 1500:
        js = sorted(set(range(1, 501)) | set(int(x) for x in np.geomspace(1, jmax, 800)))
    else:
        js = range(1, jmax + 1)
    best, kbest = -1e9, None
    for j in js:
        k = TWO_PI * j / L
        km, kp = (k - q) ** 2, (k + q) ** 2
        B = np.diag([Gamma + 2j * w + km, Gamma - 2j * w + kp])
        C = np.array([[km * (1 + 2 * A2 + 1j * (w - alpha)) + 1j * w * Gamma - w * w, A2 * km],
                      [A2 * kp, kp * (1 + 2 * A2 + 1j * (alpha - w)) - w * w - 1j * Gamma * w]])
        M = np.block([[np.zeros((2, 2)), np.eye(2)], [-C, -B]])
        lam = float(np.linalg.eigvals(M).real.max())
        if lam > best:
            best, kbest = lam, k
    return best, kbest


def pick_n(alpha, Gamma, L, tol=1e-6):
    """Γ>0 用: q≈√Γ に近い順に、二次安定な進行波になる n を探す（見つからなければ最も近い n）。"""
    nmax = max(1, int(L / 4.0))
    cands = sorted(range(1, nmax + 1), key=lambda n: abs(TWO_PI * n / L - math.sqrt(Gamma)))
    for n in cands[:60]:
        lam, _ = secondary_growth(TWO_PI * n / L, alpha, Gamma, L)
        if lam is not None and lam <= tol:
            return n
    return cands[0]


# ------------------------------------------------------------------
#  設計リゾルバ（制約伝播 → 既定値補完 → 整合性検査）
# ------------------------------------------------------------------
def propagate(v, src):
    """既知の量から厳密関係式で決まる量を埋める。固定点まで繰り返す。"""

    def setv(k, val, how):
        if k in v:
            return False
        v[k] = val
        src[k] = how
        return True

    changed = True
    while changed:
        changed = False
        # R1: q = 2πn/L
        if "n" in v and "L" in v:
            changed |= setv("q", TWO_PI * v["n"] / v["L"], "厳密関係 q=2πn/L")
        if "q" in v and "L" in v:
            changed |= setv("n", v["q"] * v["L"] / TWO_PI, "厳密関係 n=qL/2π")
        if "q" in v and "n" in v:
            changed |= setv("L", TWO_PI * v["n"] / v["q"], "厳密関係 L=2πn/q")
        # R2: ω = αq²/(Γ+q²)
        if v.get("gamma") == 0.0:
            if "alpha" in v:
                changed |= setv("omega", v["alpha"], "厳密関係 Γ=0 → ω=α")
            if "omega" in v:
                changed |= setv("alpha", v["omega"], "厳密関係 Γ=0 → α=ω")
        elif "gamma" in v:
            G = v["gamma"]
            if "alpha" in v and "q" in v:
                changed |= setv("omega", v["alpha"] * v["q"] ** 2 / (G + v["q"] ** 2), "厳密関係 ω=αq²/(Γ+q²)")
            if "omega" in v and "q" in v:
                changed |= setv("alpha", v["omega"] * (G + v["q"] ** 2) / v["q"] ** 2, "厳密関係 α=ω(Γ+q²)/q²")
            if "omega" in v and "alpha" in v and v["alpha"] > v["omega"]:
                changed |= setv("q", math.sqrt(v["omega"] * G / (v["alpha"] - v["omega"])), "厳密関係 q²=ωΓ/(α-ω)")
        elif all(k in v for k in ("omega", "alpha", "q")):
            G = v["alpha"] * v["q"] ** 2 / v["omega"] - v["q"] ** 2
            if G >= -1e-12:
                changed |= setv("gamma", max(G, 0.0), "厳密関係 Γ=αq²/ω-q²")
        # R3: A² = (ω/q)² - 1
        if "omega" in v and "q" in v:
            r = (v["omega"] / v["q"]) ** 2 - 1.0
            if r <= 0:
                raise ValueError(f"ω={v['omega']:.4g} ≤ q={v['q']:.4g}: 進行波の振幅 A²={r:.3g} が負（この組み合わせでは時間結晶状態が存在しない）")
            changed |= setv("amp", math.sqrt(r), "厳密関係 A²=(ω/q)²-1")
        if "amp" in v and "q" in v:
            changed |= setv("omega", v["q"] * math.sqrt(1.0 + v["amp"] ** 2), "厳密関係 ω=q√(1+A²)")
        if "amp" in v and "omega" in v:
            changed |= setv("q", v["omega"] / math.sqrt(1.0 + v["amp"] ** 2), "厳密関係 q=ω/√(1+A²)")
        # R4: (α, Γ>0, A) → q（2次方程式 c q² - α q + cΓ = 0, c=√(1+A²)）
        if ("alpha" in v and v.get("gamma", 0.0) > 0 and "amp" in v and "q" not in v and "omega" not in v):
            c = math.sqrt(1.0 + v["amp"] ** 2)
            disc = v["alpha"] ** 2 - 4.0 * c * c * v["gamma"]
            if disc < 0:
                raise ValueError(f"α={v['alpha']:.4g}, Γ={v['gamma']:.4g} では A={v['amp']:.4g} の進行波が存在しない"
                                 f"（α ≥ 2cΓ^½ = {2 * c * math.sqrt(v['gamma']):.4g} が必要）")
            roots = [(v["alpha"] - math.sqrt(disc)) / (2 * c), (v["alpha"] + math.sqrt(disc)) / (2 * c)]
            qq = min(roots, key=lambda r_: abs(r_ - math.sqrt(v["gamma"])))
            changed |= setv("q", qq, "厳密関係（2次方程式の根のうち q≈√Γ に近い方）")


def autofill(spec):
    """指定された量から全パラメータを決める。戻り値 (値, 由来)。spec が空でも既定値だけで決まる。"""
    v = dict(spec)
    src = {k: "指定" for k in spec}
    propagate(v, src)

    def default(k, val, why):
        if k not in v:
            v[k] = val
            src[k] = "既定: " + why
            propagate(v, src)

    def default_n():
        if "n" in v:
            return
        if "L" in v and "q" not in v:
            G = v.get("gamma", 0.0)
            if G == 0:
                default("n", 1.0, "Γ=0: 粗化は最低モード n=1 まで進む（n>1 は二次不安定）")
            elif "alpha" in v:
                default("n", float(pick_n(v["alpha"], G, v["L"])), "Γ>0: q≈√Γ に最も近い二次安定なモード")
        elif "q" in v and "L" not in v:
            default("n", 1.0, "最低モード n=1（L は q から決まる）")

    omega_alpha_differ = ("omega" in v and "alpha" in v and abs(v["omega"] - v["alpha"]) > REL_TOL * max(1.0, abs(v["alpha"])))
    if "gamma" not in v and not omega_alpha_differ:
        default("gamma", 0.0, "摩擦なし")
    default_n()
    if "alpha" not in v:
        G = v.get("gamma", 0.0)
        default("alpha", max(1.0, 3.0 * math.sqrt(G)), "α=max(1, 3√Γ)（閾値 2√Γ より十分上）")
    default_n()
    if "L" not in v and "q" not in v:
        L_def, why = 200.0, "系長 200"
        # ω が小さいと L=200 では ω ≤ q（進行波が存在しない/弱い）→ A=1 となる L に取り直す
        if "omega" in v and v.get("gamma") == 0.0 and "n" not in v:
            if v["omega"] < math.sqrt(2.0) * TWO_PI / L_def:
                L_def = TWO_PI * math.sqrt(2.0) / v["omega"]
                why = "ω が小さく L=200 では進行波が存在しない/弱い → A=1 になる L"
        default("L", L_def, why)
        default_n()

    missing = [k for k in ("alpha", "gamma", "L", "n", "q", "omega", "amp") if k not in v]
    if missing:
        raise ValueError(f"決められませんでした: {missing}。指定を増やしてください。")
    verify(v)
    return v, src


def verify(v):
    """指定の過剰・矛盾と n の整数性を検査する。"""
    n = v["n"]
    if abs(n - round(n)) > 1e-6 * max(1.0, abs(n)):
        cands = sorted({max(1, math.floor(n)), max(1, math.ceil(n))})
        sug = ", ".join(f"n={c} → L={TWO_PI * c / v['q']:.4g}" for c in cands)
        raise ValueError(f"波の山の数 n={n:.4f} が整数になりません（q={v['q']:.4g}, L={v['L']:.4g}）。"
                         f"L か目標値を調整してください。候補: {sug}")
    v["n"] = float(round(n))

    def rel(a, b):
        return abs(a - b) / max(1e-12, abs(a), abs(b))

    checks = [("q=2πn/L", v["q"], TWO_PI * v["n"] / v["L"]),
              ("ω=αq²/(Γ+q²)", v["omega"], v["alpha"] * v["q"] ** 2 / (v["gamma"] + v["q"] ** 2)),
              ("A²=(ω/q)²-1", v["amp"] ** 2, (v["omega"] / v["q"]) ** 2 - 1.0)]
    for name, a, b in checks:
        if rel(a, b) > 1e-6:
            raise ValueError(f"指定値が厳密関係 {name} と矛盾します（{a:.6g} vs {b:.6g}）。指定を減らしてください。")
    if v["gamma"] < 0 or v["alpha"] <= 0 or v["L"] <= 0:
        raise ValueError("α>0, Γ≥0, L>0 が必要です。")


# ------------------------------------------------------------------
#  入力（倍率 / 持続時間 / 持続周期）の解決
# ------------------------------------------------------------------
def resolve_timing(ratio, duration, cycles, Tin):
    """
    指定された (m, D, P) から応答周期 T_resp が決まるなら ω の指定を返す（決まらなければ空）。
      m      → T_resp = m·T_in
      D & P  → T_resp = D/P
    両方で決まる場合は互いに一致していなければエラー。
    """
    cand = []
    if ratio is not None:
        cand.append(("倍率 × 入力周期 (m·T_in)", ratio * Tin))
    if duration is not None and cycles is not None:
        cand.append(("持続時間 ÷ 持続周期 (D/P)", duration / cycles))
    if len(cand) == 2:
        (na, a), (nb, b) = cand
        if abs(a - b) / max(a, b) > 1e-3:
            raise ValueError(f"指定が矛盾しています: {na} = {a:.6g} だが {nb} = {b:.6g}。"
                             "3つ指定する場合は m·T_in = D/P となるようにしてください（どれか1つで足ります）。")
    return {"omega": TWO_PI / cand[0][1]} if cand else {}


def build_timing(d, ratio, duration, cycles, Tin, scratch):
    """設計点 d から (T_in, T_resp, m, D, P) とその由来を作る。"""
    Tr = TWO_PI / d["omega"]
    tm = {"Tin": Tin, "Tr": Tr, "m": Tr / Tin}
    ts = {"Tin": "基準（--Tin, 既定 1）", "Tr": "導出 T_resp=2π/ω"}
    ts["m"] = "指定" if ratio is not None else "導出 m=T_resp/T_in"
    if duration is not None:
        tm["D"], ts["D"] = duration, "指定"
    elif cycles is not None:
        tm["D"], ts["D"] = cycles * Tr, "導出 D=P×T_resp"
    else:
        if scratch:
            tm["D"], ts["D"] = 3000.0, "既定: 自己組織化の目安 3000"
        else:
            tm["D"], ts["D"] = max(300.0, 40.0 * Tr), "既定: max(300, 40 周期)"
    if cycles is not None:
        tm["P"], ts["P"] = cycles, "指定"
    else:
        tm["P"], ts["P"] = tm["D"] / Tr, "導出 P=D/T_resp"
    return tm, ts


# ------------------------------------------------------------------
#  数値計算コア（RK4, 周期境界, dt=0.01 固定）
# ------------------------------------------------------------------
def laplacian(f, dx):
    out = np.empty_like(f)
    out[1:-1] = f[2:] + f[:-2]
    out[0] = f[1] + f[-1]
    out[-1] = f[-2] + f[0]
    out -= 2.0 * f
    out *= 1.0 / (dx * dx)
    return out


def rhs(phi, psi, alpha, Gamma, dx):
    bracket = (1.0 - 1j * alpha) * phi + psi + np.abs(phi) ** 2 * phi
    return psi, laplacian(bracket, dx) - Gamma * psi


def rk4_step(phi, psi, dt, alpha, Gamma, dx):
    k1p, k1s = rhs(phi, psi, alpha, Gamma, dx)
    k2p, k2s = rhs(phi + 0.5 * dt * k1p, psi + 0.5 * dt * k1s, alpha, Gamma, dx)
    k3p, k3s = rhs(phi + 0.5 * dt * k2p, psi + 0.5 * dt * k2s, alpha, Gamma, dx)
    k4p, k4s = rhs(phi + dt * k3p, psi + dt * k3s, alpha, Gamma, dx)
    return (phi + dt / 6.0 * (k1p + 2 * k2p + 2 * k3p + k4p),
            psi + dt / 6.0 * (k1s + 2 * k2s + 2 * k3s + k4s))


def dt_limit(max_abs_phi, alpha, dx, cfl=CFL):
    """振幅で変わる有効波速 c≈√(|1-iα|+3|φ|²max) から決まる、その瞬間に許される dt の目安。"""
    c = math.sqrt(math.hypot(1.0, alpha) + 3.0 * float(max_abs_phi) ** 2)
    return min(cfl / (2.0 * c / dx), 0.5 * dx * dx)


def stationarity(a_win, dt_rec, qn=None, alpha=None, Gamma=None, dx=None):
    """
    直近窓での定常性。次をすべて満たすと時間結晶状態（定常）とみなす:
      支配モード一定 / 振幅変動<2% / ω変動<5% / 純度>98% /
      同じ波数の厳密進行波(S24)に対し ω が 2%, A が 3% 以内（qn,alpha,Gamma,dx を渡したとき）
    """
    amp = np.abs(a_win)
    dom_each = amp.argmax(axis=1)
    dom = int(dom_each[-1])
    same = bool(np.all(dom_each == dom))
    x = amp[:, dom]
    amp_cv = float(x.std() / x.mean()) if x.mean() > 0 else float("inf")
    th = np.unwrap(np.angle(a_win[:, dom]))
    om = np.diff(th) / dt_rec
    om_mean = float(abs(om.mean()))
    om_cv = float(om.std() / om_mean) if om_mean > 1e-12 else float("inf")
    purity = float(np.mean(amp[:, dom] ** 2 / np.maximum((amp ** 2).sum(axis=1), 1e-300)))
    exact_ok = True
    if qn is not None:
        q_dom = abs(float(qn[dom]))
        if q_dom <= 0:
            exact_ok = False
        else:
            w_p, A_p = travelling_wave(q_discrete(q_dom, dx), alpha, Gamma)
            exact_ok = (A_p is not None and abs(om_mean / w_p - 1) < 0.02 and abs(float(x.mean()) / A_p - 1) < 0.03)
    return {"dom": dom, "same": same, "amp": float(x.mean()), "amp_cv": amp_cv,
            "omega": om_mean, "om_cv": om_cv, "purity": purity, "exact_ok": exact_ok,
            "stationary": same and amp_cv < 0.02 and om_cv < 0.05 and purity > 0.98 and exact_ok}


# ------------------------------------------------------------------
#  実行計画（dt 固定 → dx を自動調整）
# ------------------------------------------------------------------
def plan_run(d, T_req, mode, dx_target):
    """dt=0.01 固定のもとで CFL を満たす格子・記録間隔・総ステップ数を決める。"""
    alpha, Gamma, L, n, q = d["alpha"], d["gamma"], d["L"], int(d["n"]), d["q"]
    w, A = d["omega"], d["amp"]
    Tr = TWO_PI / w
    # 遷移中の振幅の余裕（check: 1.3A, scratch: 1.6A）を見込んだ最小 dx
    A_eff = A * (1.3 if mode == "check" else 1.6)
    c = math.sqrt(math.hypot(1.0, alpha) + 3.0 * A_eff**2)
    dx_cfl = 2.0 * c * DT / CFL
    dx_visc = math.sqrt(DT / 0.5)
    dx_min = max(dx_cfl, dx_visc)
    dx_req = max(dx_target, dx_min)
    N = 2 * int(L / dx_req / 2.0)
    grid_ok = N >= 16
    if not grid_ok:
        N = 16
    dx = L / N
    qe = q_discrete(q, dx)
    w_e, A_e = travelling_wave(qe, alpha, Gamma)

    spr = max(1, int(round(min(100.0, 0.5 / w / DT))))   # 記録1回あたりのステップ数
    dt_rec = spr * DT
    T_sim = max(T_req, 3.0 * Tr)                         # 判定には最低3周期必要
    nrec = int(math.ceil(T_sim / dt_rec - 1e-9))
    nwin = max(40, int(math.ceil(5.0 * Tr / dt_rec)))
    nmax = min(N // 2 - 1, 256)
    modes = np.r_[0:nmax + 1, N - nmax:N]
    return {"N": N, "dx": dx, "dx_min": dx_min, "dx_req": dx_req, "grid_ok": grid_ok,
            "qe": qe, "w_e": w_e, "A_e": A_e, "qdx": q * dx,
            "spr": spr, "dt_rec": dt_rec, "T_sim": nrec * dt_rec, "T_req": T_req, "nrec": nrec,
            "nwin": nwin, "modes": modes, "steps_per_period": Tr / DT, "extended": T_sim > T_req + 1e-9}


def init_state(d, plan, mode, seed, noise):
    alpha, q = d["alpha"], d["q"]
    N, dx = plan["N"], plan["dx"]
    w_e, A_e = plan["w_e"], plan["A_e"]
    x = np.arange(N) * dx
    rng = np.random.default_rng(seed)
    if mode == "check":
        base = A_e * np.exp(-1j * q * x)
        psi = 1j * w_e * base
        phi = base + noise * A_e * (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / math.sqrt(2.0)
    else:
        phi = 0.05 * (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / math.sqrt(2.0)
        psi = np.zeros(N, dtype=complex)
    return phi, psi


def benchmark(d, plan, mode, seed, noise):
    """実際の格子で数サイクル回し、記録1回（spr ステップ + 記録）あたりの実時間[秒]を測る。"""
    alpha, Gamma = d["alpha"], d["gamma"]
    N, dx, spr, modes = plan["N"], plan["dx"], plan["spr"], plan["modes"]
    phi, psi = init_state(d, plan, mode, seed, noise)
    cycles = 0
    t0 = time.perf_counter()
    with np.errstate(all="ignore"):
        while True:
            for _ in range(spr):
                phi, psi = rk4_step(phi, psi, DT, alpha, Gamma, dx)
            _ = (np.fft.fft(phi) / N)[modes]
            _ = float(np.max(np.abs(phi)))
            cycles += 1
            el = time.perf_counter() - t0
            if (cycles >= 3 and el > 0.25) or el > 2.0 or cycles >= 60:
                break
    return el / cycles


# ------------------------------------------------------------------
#  完成可能性の事前判定
# ------------------------------------------------------------------
def fmt_dur(sec):
    """h:mm:ss（1日以上は 'N日 h:mm:ss'）。"""
    if sec is None or not math.isfinite(sec):
        return "--:--:--"
    if sec > 999 * 86400:
        return ">999日"
    s = int(round(max(sec, 0.0)))
    days, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return (f"{days}日 " if days else "") + f"{h:d}:{m:02d}:{s:02d}"


def fmt_human(sec):
    if sec < 1:
        return f"{sec * 1000:.0f} ms"
    if sec < 60:
        return f"{sec:.1f} 秒"
    if sec < 3600:
        return f"{sec / 60:.1f} 分"
    if sec < 86400:
        return f"{sec / 3600:.1f} 時間"
    return f"{sec / 86400:.1f} 日"


def assess(d, tm, plan, mode, noise, est_total_s, max_hours):
    """
    完成可能性を評価。戻り値 (総合レベル 0-3, 項目リスト, 実行中止すべきか)。
    項目 = (名前, レベル, 説明, blocking)。blocking かつ「極めて低い」なら既定では実行しない。
    """
    items = []

    def add(name, lvl, msg, block=False):
        items.append((name, lvl, msg, block))

    # 1) 存在
    if plan["A_e"] is None:
        add("存在", 3, "格子の離散化後は厳密進行波が存在しない（q·dx が大きすぎる）", True)
    else:
        add("存在", 0, f"厳密進行波が存在する（A={d['amp']:.4g}, 格子上では A={plan['A_e']:.4g}）")
    if mode == "scratch" and band(d["alpha"], d["gamma"]) is None:
        add("自己組織化", 3, "α ≤ 2√Γ で線形不安定帯がなく、ランダム初期条件からは時間結晶状態に育たない", True)

    # 2) dt=0.01 固定での数値安定性・精度
    S = plan["steps_per_period"]
    lvl = 3 if S < 8 else (2 if S < 25 else 0)
    add("時間刻み dt=0.01", lvl,
        f"応答1周期 = {S:.0f} ステップ" + ("（少なすぎて位相誤差が大きい）" if lvl else "（十分）"), lvl == 3)
    if not plan["grid_ok"]:
        add("空間格子", 3,
            f"dt=0.01 固定で CFL を満たすには dx≥{plan['dx_req']:.3g} が必要だが、系長 L={d['L']:.4g} では格子点が 16 個に満たない", True)
    else:
        qdx = plan["qdx"]
        lvl = 3 if qdx > 2.0 else (2 if qdx > 1.0 else 0)
        add("空間格子", lvl,
            f"dx={plan['dx']:.4g}, N={plan['N']}（dt=0.01 の CFL 下限 dx≥{plan['dx_min']:.3g}）, 1波長あたり {plan['N'] / d['n']:.1f} 点"
            + ("" if lvl == 0 else "（波長に対して粗すぎ、精度が落ちる）"), lvl == 3)
    if d["gamma"] * DT > 2.0:
        add("摩擦項", 3, f"Γ·dt={d['gamma'] * DT:.3g} > 2（dt=0.01 では摩擦項が不安定）", True)

    # 3) 持続（二次安定性）
    lam = d.get("lam")
    D = tm["D"]
    if lam is None:
        add("持続（二次安定性）", 1, "二次安定性を評価できなかった")
    elif lam <= 1e-4:
        add("持続（二次安定性）", 0, f"解析的に二次安定（最大成長率 {lam:+.1e} ≈ 0）。設計状態は保たれる見込み")
    else:
        if mode == "check":
            ts = math.log(1.0 / max(noise, 1e-12)) / lam
            if D >= 3.0 * ts:
                lvl = 3
            elif D > ts:
                lvl = 2
            else:
                lvl = 1
            add("持続（二次安定性）", lvl,
                f"二次不安定 λ={lam:.3g}（e倍時間 {1 / lam:.3g}）。ノイズ {noise * 100:g}% で崩れるのは t≈{ts:.3g}"
                f" ＝ 要求 D={D:.4g} の {ts / D * 100:.1f}%")
        else:
            add("持続（二次安定性）", 3,
                f"設計状態は二次不安定（λ={lam:.3g}）。自己組織化しても設計状態は保たれない")
    if mode == "scratch" and D < 1500:
        add("自己組織化の時間", 2, f"粗化には通常 t~数千が必要。D={D:.4g} では不足の可能性が高い")

    # 4) 計算時間
    if est_total_s is not None:
        hours = est_total_s / 3600.0
        if hours > max_hours:
            lvl = 3
        elif hours > max_hours / 6.0:
            lvl = 2
        elif est_total_s > 600.0:
            lvl = 1
        else:
            lvl = 0
        msg = f"推定所要 {fmt_human(est_total_s)}（{plan['nrec'] * plan['spr']:,} ステップ, 上限の目安 {max_hours:g} 時間）"
        if lvl == 3:
            msg += " → 現実的な時間で終わらない"
        add("計算時間", lvl, msg, lvl == 3)

    overall = max(it[1] for it in items)
    blocked = any(it[3] and it[1] == 3 for it in items)
    return overall, items, blocked


def print_assessment(overall, items, blocked, forced):
    print("\n--- 完成可能性の事前判定 ---")
    for name, lvl, msg, _ in items:
        mark = ["  OK ", " 注意 ", " 低い ", "極低い"][lvl]
        print(f"  [{mark}] {name:<14} {msg}")
    if overall == 3:
        print("\n  ************************************************************")
        print("  *   完成可能性: 極めて低い                                  *")
        print("  ************************************************************")
        for name, lvl, msg, _ in items:
            if lvl == 3:
                print(f"    理由: {name} — {msg}")
        if blocked and not forced:
            print("    → 実行を中止します（それでも回すなら --force）。")
        elif blocked:
            print("    → --force 指定のため実行しますが、結果は信頼できません。")
        else:
            print("    → 数値的には回せるため、実測で確認します。")
    else:
        print(f"\n  総合: 完成可能性 {LEVELS[overall]}")


# ------------------------------------------------------------------
#  進捗・残り時間表示
# ------------------------------------------------------------------
class Progress:
    """進捗%・経過・残り時間・終了予定時刻を表示。ベンチ値で初期化し、実測速度(EMA)で更新する。"""

    def __init__(self, total, prior_cycle_s, T_sim, spr):
        self.total, self.T_sim, self.spr = total, T_sim, spr
        self.t0 = self.last_t = time.perf_counter()
        self.last_k = 0
        self.rate = (1.0 / prior_cycle_s) if prior_cycle_s and prior_cycle_s > 0 else None   # 記録/秒
        self.measured = False
        self.tty = sys.stderr.isatty()
        self.interval = 0.5 if self.tty else 5.0
        self.width = 0

    def update(self, k, extra="", force=False):
        now = time.perf_counter()
        if not force and now - self.last_t < self.interval:
            return
        dk, dtw = k - self.last_k, now - self.last_t
        if dtw > 0 and dk > 0:
            inst = dk / dtw
            self.rate = inst if (self.rate is None or not self.measured) else 0.3 * inst + 0.7 * self.rate
            self.measured = True
            self.last_t, self.last_k = now, k
        el = now - self.t0
        frac = min(1.0, k / self.total)
        eta = (self.total - k) / self.rate if self.rate else None
        bar = "█" * int(frac * 20) + "░" * (20 - int(frac * 20))
        fin = time.strftime("%m/%d %H:%M:%S", time.localtime(time.time() + eta)) if eta is not None and eta < 999 * 86400 else "--"
        sps = f"{self.rate * self.spr:,.0f} step/s" if self.rate else "-- step/s"
        line = (f"[{bar}] {frac * 100:5.1f}%  t={k * self.spr * DT:.5g}/{self.T_sim:.5g}  "
                f"経過 {fmt_dur(el)}  残り {fmt_dur(eta)}{'' if self.measured else '(見積)'}  終了予定 {fin}  {sps}  {extra}")
        if self.tty:
            self.width = max(self.width, len(line))
            sys.stderr.write("\r" + line.ljust(self.width))
        else:
            sys.stderr.write(line + "\n")
        sys.stderr.flush()

    def done(self, k, note=""):
        el = time.perf_counter() - self.t0
        if self.tty:
            sys.stderr.write("\r" + " " * self.width + "\r")
        sys.stderr.write(f"完了: 実所要 {fmt_dur(el)}（{fmt_human(el)}）{note}\n")
        sys.stderr.flush()
        return el


# ------------------------------------------------------------------
#  シミュレーション本体
# ------------------------------------------------------------------
def simulate(d, plan, mode, seed, noise, est_cycle_s=None):
    alpha, Gamma, L, n = d["alpha"], d["gamma"], d["L"], int(d["n"])
    N, dx, spr, nrec, nwin = plan["N"], plan["dx"], plan["spr"], plan["nrec"], plan["nwin"]
    dt_rec, modes = plan["dt_rec"], plan["modes"]
    phi, psi = init_state(d, plan, mode, seed, noise)
    qn = TWO_PI * np.fft.fftfreq(N, d=dx)[modes]

    win = deque(maxlen=nwin)
    hstride = max(1, nrec // 4000)
    rstride = max(1, nrec // 800)
    chk = max(1, nwin // 2)
    t_h, rms_h, re_h, mon = [], [], [], []
    t_alive, bad, collapsed, stopped = 0.0, 0, False, False

    def record(k):
        a = (np.fft.fft(phi) / N)[modes]
        ab = np.abs(phi)
        win.append(a)
        if k % hstride == 0:
            t_h.append(k * dt_rec)
            rms_h.append(math.sqrt(float(np.mean(ab * ab))))
        if k % rstride == 0:
            re_h.append((k * dt_rec, phi.real.astype(np.float32)))
        return float(ab.max())

    record(0)
    prog = Progress(nrec, est_cycle_s, plan["T_sim"], spr)
    prog.update(0, force=True)
    k_done = 0
    for k in range(1, nrec + 1):
        for _ in range(spr):
            phi, psi = rk4_step(phi, psi, DT, alpha, Gamma, dx)
        t = k * dt_rec
        if not np.isfinite(phi).all():
            prog.done(k)
            raise FloatingPointError(f"発散しました (t={t:.1f})。dt=0.01 固定では数値的に保てませんでした。")
        mx = record(k)
        k_done = k
        lim = dt_limit(mx, alpha, dx)
        if DT > 1.3 * lim:
            prog.done(k)
            raise FloatingPointError(f"t={t:.1f}: 振幅が育ち dt=0.01 では CFL を満たせなくなりました（許容 dt≈{lim:.4f}）。")
        if k >= nwin and k % chk == 0:
            st = stationarity(np.array(win), dt_rec, qn, alpha, Gamma, dx)
            n_dom = int(round(abs(float(qn[st["dom"]])) * L / TWO_PI))
            mon.append((t, st["omega"], st["amp"], n_dom, st["stationary"]))
            if mode == "check":
                if st["stationary"] and n_dom == n:
                    t_alive, bad = t, 0
                else:
                    bad += 1
                    if bad >= 2:
                        collapsed = True
                        prog.update(k, f"|a|={st['amp']:.3g}", force=True)
                        break
            elif st["stationary"]:
                stopped = True
                prog.update(k, f"|a|={st['amp']:.3g}", force=True)
                break
        prog.update(k, f"max|φ|={mx:.3g}")
    note = "（崩壊を検出して打ち切り）" if collapsed else ("（定常に達して自動停止）" if stopped else "")
    elapsed = prog.done(k_done, note)
    return {"t_end": k_done * dt_rec, "t_h": np.array(t_h), "rms_h": np.array(rms_h), "re": re_h,
            "mon": np.array(mon) if mon else np.zeros((0, 5)), "win": np.array(win), "qn": qn,
            "N": N, "dx": dx, "dt_rec": dt_rec, "nwin": nwin, "mode": mode,
            "collapsed": collapsed, "stopped": stopped, "t_alive": t_alive, "elapsed": elapsed,
            "w_e": plan["w_e"], "A_e": plan["A_e"]}


# ------------------------------------------------------------------
#  入出力
# ------------------------------------------------------------------
LABELS = {"alpha": "α (activity)", "gamma": "Γ (摩擦)", "L": "L (系長)", "n": "n (波の山の数)",
          "q": "q (波数)", "omega": "ω (角周波数)", "amp": "A (振幅 |a_n|)"}


def print_design(d, src, tm, ts):
    print("\n--- 入力から決まった周期・持続（T_in は基準）---")
    print(f"  入力周期 T_in            = {tm['Tin']:<10.5g}  [{ts['Tin']}]")
    print(f"  応答周期 T_resp = 2π/ω   = {tm['Tr']:<10.5g}  [{ts['Tr']}]")
    print(f"  倍率 m = T_resp/T_in     = {tm['m']:<10.5g}  [{ts['m']}]")
    print(f"  持続時間 D               = {tm['D']:<10.5g}  [{ts['D']}]")
    print(f"  持続周期 P = D/T_resp    = {tm['P']:<10.5g}  [{ts['P']}]")
    print("\n--- 設計点（自動決定。入力ではなく結果）---")
    for k in ("alpha", "gamma", "L", "n", "q", "omega", "amp"):
        val = f"{int(d[k])}" if k == "n" else f"{d[k]:.5g}"
        print(f"  {LABELS[k]:<18} = {val:<10}  [{src[k]}]")
    bd = band(d["alpha"], d["gamma"])
    print(f"  線形不安定: α>2√Γ={2 * math.sqrt(d['gamma']):.4g} → " + ("成立" if bd else "不成立（自己組織化は起きない。check は可能）"))
    if bd:
        print(f"  不安定帯 ({bd[0]:.4g}, {bd[1]:.4g}) に q が入っているか: {bd[0] < d['q'] < bd[1]}")
    lam, k_l = secondary_growth(d["q"], d["alpha"], d["gamma"], d["L"])
    d["lam"] = lam
    if lam is not None:
        if lam > 1e-4:
            print(f"  二次安定性(SM S32): 不安定  最大成長率 λ={lam:.4g} (k={k_l:.3g}), e倍時間 1/λ={1 / lam:.4g}")
        else:
            print(f"  二次安定性(SM S32): 安定（箱 L で許される摂動の最大成長率 {lam:+.1e} ≈ 0）")


def print_plan(plan, mode):
    print(f"\n--- 実行計画（dt={DT} 固定, RK4, モード={mode}）---")
    print(f"  格子: dx={plan['dx']:.4g}, N={plan['N']}   （dx≥{plan['dx_min']:.3g} が dt=0.01 の CFL 下限）")
    print(f"  総ステップ {plan['nrec'] * plan['spr']:,}（記録間隔 {plan['dt_rec']:.3g} = {plan['spr']} ステップ）, シミュレーション時間 T={plan['T_sim']:.5g}")
    if plan["extended"]:
        print(f"  ※判定には最低3周期が必要なため、持続時間 D={plan['T_req']:.5g} を T={plan['T_sim']:.5g} に延長して回します。")


def analyze(res, d):
    st = stationarity(res["win"], res["dt_rec"], res["qn"], d["alpha"], d["gamma"], res["dx"])
    q_dom = abs(float(res["qn"][st["dom"]]))
    n_dom = int(round(q_dom * d["L"] / TWO_PI))
    qe_dom = q_discrete(q_dom, res["dx"]) if q_dom > 0 else 0.0
    w_p, A_p = travelling_wave(qe_dom, d["alpha"], d["gamma"]) if qe_dom > 0 else (float("nan"), None)
    return st, q_dom, n_dom, w_p, A_p


def report(res, d, tm, st, n_dom, w_p, A_p):
    mode = res["mode"]
    design_ok = (n_dom == int(d["n"]))
    alive_end = st["stationary"] and design_ok
    Tr_meas = TWO_PI / st["omega"] if st["omega"] > 1e-12 else float("inf")
    print("\n--- 検証結果 ---")
    print(f"  実行時間 t={res['t_end']:g}（実所要 {fmt_dur(res['elapsed'])}）")
    print(f"  最終状態: 支配モード n={n_dom}, ω={st['omega']:.5g}, |a|={st['amp']:.5g}"
          f"  （窓内変動: 振幅 {st['amp_cv'] * 100:.2f}%, ω {st['om_cv'] * 100:.2f}%, 支配モードの純度 {st['purity'] * 100:.1f}%）")
    if A_p is not None:
        print(f"  同じ n の厳密進行波(S24, 離散補正済み): ω={w_p:.5g}, A={A_p:.5g}  → 比 ω:{st['omega'] / w_p:.3f}, A:{st['amp'] / A_p:.3f}")
    print(f"  設計値との比（離散補正済み）: ω:{st['omega'] / res['w_e']:.3f}, A:{st['amp'] / res['A_e']:.3f}")
    print(f"  応答周期(実測) = {Tr_meas:.5g} → 入力周期の {Tr_meas / tm['Tin']:.5g} 倍（設計 {tm['m']:.5g} 倍）")

    completed = False
    if mode == "check":
        if res["collapsed"]:
            ta = res["t_alive"]
            print(f"  持続時間(実測) ≈ {ta:.5g}（{ta / tm['Tr']:.4g} 周期）… 要求 D={tm['D']:.5g}（{tm['P']:.4g} 周期）の {ta / tm['D'] * 100:.1f}% で崩壊")
            lam = d.get("lam")
            msg = "設計状態は崩れた" + (f"（解析 λ={lam:.3g} と整合）。" if lam is not None and lam > 1e-4 else "（解析では安定のはずで不一致 → 離散化や条件を要確認）。")
        elif alive_end:
            T = res["t_end"]
            print(f"  持続時間(実測) ≥ {T:.5g}（{T / tm['Tr']:.4g} 周期）… 要求 D={tm['D']:.5g}（{tm['P']:.4g} 周期）の間 保たれた")
            lam = d.get("lam")
            if lam is not None and lam > 1e-4:
                msg = f"T の範囲では保たれたが、解析的には不安定（λ={lam:.3g}, e倍時間 {1 / lam:.0f}）。さらに長く回すといずれ崩れる。"
            else:
                msg = "設計した時間結晶状態は保たれた（解析的にも二次安定、数値でも確認）。"
                completed = True
        else:
            print("  持続時間(実測): 終了時点で設計状態の基準を満たさない")
            msg = "終了時点で設計状態と判定できない（判定窓が短い / 時間が足りない可能性。持続周期を増やして再実行）。"
    else:
        if st["stationary"] and design_ok:
            msg = "ランダム初期条件から、設計した時間結晶状態に自己組織化した。"
            completed = True
        elif st["stationary"]:
            msg = f"ランダム初期条件から n={n_dom} の定常状態（時間結晶状態）に落ち着いたが、設計値とは別の状態。"
        else:
            msg = "定常に達していない（粗化が進行中、または D が不足）。"
    print("  判定:", msg)
    print("  最終:", "★ 要求した持続時間の間、時間結晶状態を確認できた" if completed else "✕ 要求どおりの完成は確認できなかった")
    print("  ※1D・単一シード・有限時間・簡易基準（支配モード一定/振幅変動<2%/ω変動<5%/純度>98%/厳密進行波S24との一致 ω 2%・A 3%）での判定です。")
    return completed


def make_plot(res, d, tm, st, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(15, 8))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    ax = fig.add_subplot(gs[0, :])
    if len(res["re"]) >= 2:
        tt = np.array([r[0] for r in res["re"]])
        re = np.array([r[1] for r in res["re"]])
        vmax = float(np.max(np.abs(re)))
        im = ax.imshow(re.T, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       extent=[tt[0], tt[-1], 0, d["L"]])
        plt.colorbar(im, ax=ax, pad=0.01)
    ax.set_xlabel("t"); ax.set_ylabel("x"); ax.set_title("Re φ")

    mon = res["mon"]
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(res["t_h"], res["rms_h"], "c-", alpha=0.7, label="rms |φ|")
    if len(mon):
        ax.plot(mon[:, 0], mon[:, 2], "m.-", label="|a_dom| (window)")
    ax.axhline(d["amp"], color="k", ls="--", label=f"design A={d['amp']:.3g}")
    ax.set_xlabel("t"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_title("Amplitude")

    ax = fig.add_subplot(gs[1, 1])
    if len(mon):
        ax.plot(mon[:, 0], mon[:, 1], "g.-", label="ω of dominant mode (window)")
    ax.axhline(d["omega"], color="k", ls="--", label=f"design ω={d['omega']:.3g}")
    ax.set_xlabel("t"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_title("Frequency")

    ax = fig.add_subplot(gs[1, 2])
    qq = np.linspace(0, max(1.2 * d["alpha"], 1.5 * d["q"], 0.5), 400)
    ax.plot(qq, sigma_plus(qq, d["alpha"], d["gamma"]).real, "r-")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(d["q"], color="b", ls="--", label=f"design q={d['q']:.3g}")
    ax.set_xlabel("q"); ax.set_ylabel("Re σ₊(q)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_title("Linear growth rate (S22)")
    fig.suptitle(f"φ model | m=T_resp/T_in={tm['m']:.4g}, P={tm['P']:.4g} cycles | α={d['alpha']:.3g}, Γ={d['gamma']:.3g}, "
                 f"L={d['L']:.4g}, n={int(d['n'])} | dt={DT} | mode={res['mode']}", fontweight="bold")
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n図を保存: {path}")


def interactive():
    print("次のうち、どれか1つ以上を入力してください（空欄=自動。残りは自動で決まり結果として表示されます）")
    vals = []
    for label in ("応答周期は入力周期の何倍か", "持続時間", "持続周期（応答の何周期ぶん持続させるか）"):
        s = input(f"  {label} [空欄=自動]: ").strip()
        vals.append(float(s) if s else None)
    scratch = input("  ランダム初期条件から自己組織化を見ますか? [y/N]: ").strip().lower() == "y"
    return vals[0], vals[1], vals[2], scratch


def main():
    interactive_mode = len(sys.argv) == 1 and sys.stdin.isatty()
    ap = argparse.ArgumentParser(description="φ model 時間結晶シミュレータ v5（倍率 / 持続時間 / 持続周期のどれか1つで回る, dt=0.01 固定）")
    ap.add_argument("--ratio", type=float, default=None, help="応答周期は入力周期の何倍か（m = T_resp/T_in）")
    ap.add_argument("--duration", type=float, default=None, help="持続時間 D（無次元時間）")
    ap.add_argument("--cycles", type=float, default=None, help="持続周期 P（応答の何周期ぶん持続させるか）")
    ap.add_argument("--Tin", type=float, default=1.0, help="基準となる入力周期（無次元, 既定 1）")
    ap.add_argument("--scratch", action="store_true", help="ランダム初期条件から自己組織化を見る（既定: 設計した進行波＋ノイズから持続を見る）")
    ap.add_argument("--noise", type=float, default=0.01, help="check の初期ノイズ（A に対する割合）")
    ap.add_argument("--dx", type=float, default=1.0, help="目標の空間刻み（dt=0.01 の CFL に足りなければ自動で粗くする）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-hours", type=float, default=6.0, help="計算時間の上限の目安[時間]。推定がこれを超えると『極めて低い』（既定 6）")
    ap.add_argument("--force", action="store_true", help="完成可能性が極めて低くても実行する")
    ap.add_argument("--out", default="dtc_phi_design.png")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    ratio, duration, cycles = args.ratio, args.duration, args.cycles
    if interactive_mode:
        ratio, duration, cycles, args.scratch = interactive()
    if ratio is None and duration is None and cycles is None:
        ap.error("--ratio（応答/入力周期の倍率）, --duration（持続時間）, --cycles（持続周期）のうち少なくとも1つを指定してください")
    for name, val in (("--ratio", ratio), ("--duration", duration), ("--cycles", cycles), ("--Tin", args.Tin)):
        if val is not None and not (val > 0 and math.isfinite(val)):
            ap.error(f"{name} は正の有限な数にしてください")

    print("=" * 72)
    print(" φ model 時間結晶シミュレータ v5   ∇²[(1-iα)φ+φ̇+|φ|²φ] = φ̈ + Γφ̇   dt=0.01 固定")
    print("=" * 72)
    try:
        spec = resolve_timing(ratio, duration, cycles, args.Tin)
    except ValueError as e:
        print(f"\n[入力エラー] {e}")
        return 2
    try:
        d, src = autofill(spec)
    except ValueError as e:
        print(f"\n[設計不可] {e}")
        print("\n  完成可能性: 極めて低い（この入力では時間結晶状態の設計点が作れません）")
        return 2
    if "omega" in spec:
        src["omega"] = "入力から決定 ω=2π/応答周期"
    mode = "scratch" if args.scratch else "check"
    tm, ts = build_timing(d, ratio, duration, cycles, args.Tin, args.scratch)
    print_design(d, src, tm, ts)

    plan = plan_run(d, tm["D"], mode, args.dx)
    print_plan(plan, mode)
    print("\n計算速度を計測中...", end="", flush=True)
    cyc_s = benchmark(d, plan, mode, args.seed, args.noise)
    est_total = cyc_s * plan["nrec"]
    print(f" 完了（{plan['spr'] / cyc_s:,.0f} step/s → 全体で最大 {fmt_human(est_total)} の見積もり）")

    overall, items, blocked = assess(d, tm, plan, mode, args.noise, est_total, args.max_hours)
    print_assessment(overall, items, blocked, args.force)
    if blocked and not args.force:
        if interactive_mode and input("\n  それでも実行しますか? [y/N]: ").strip().lower() == "y":
            pass
        else:
            return 4

    print()
    try:
        res = simulate(d, plan, mode, args.seed, args.noise, est_cycle_s=cyc_s)
    except FloatingPointError as e:
        print(f"\n[実行エラー] {e}")
        print("  完成可能性: 極めて低い（この条件では dt=0.01 固定のまま数値計算を保てません）")
        return 3
    except KeyboardInterrupt:
        print("\n[中断] ユーザーにより中断されました。")
        return 130
    st, q_dom, n_dom, w_p, A_p = analyze(res, d)
    report(res, d, tm, st, n_dom, w_p, A_p)
    if not args.no_plot:
        try:
            make_plot(res, d, tm, st, args.out)
        except ImportError:
            print("\n[注意] matplotlib が無いため図は保存しません（pip install matplotlib）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
