
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence
import math
import numpy as np


@dataclass(frozen=True)
class Term:
    label: str
    B: float
    C: float
    omega: float


TERMS = (Term('Z', .3, .1, 9.8),)
TERM_BY_LABEL = {x.label:x for x in TERMS}
DEFAULT_INPUT = dict(
    B_hat=.29999229150696777,
    C_hat=.09857382663171949,
    epsilon_B=.0006990334755408736,
    epsilon_C=.0028770334755408734,
    omega_hat=9.73046875,
    frequency_error_bound=.09987269585885694,
    omega_min=8.5,
    omega_max=11.0,
)
DEFAULT_INPUTS = {'Z':DEFAULT_INPUT}


@dataclass(frozen=True)
class Config:
    target_B_error: float = 1e-4       # 要求返回区间相对于估计中心的最大半径
    target_C_error: float = 1e-4
    max_B_time: float = 1024.0        # 最长单次询问时间，不是总实验时间
    max_C_time: float = 64.0
    max_C_model_phase_error: float = .20  # theta 的误差预算（不是 2*theta）
    shots: Optional[int] = None       # 例如 200000；每个时间、每个正交测量的次数
    measurement_failure_probability: float = .01  # B、C 全部精化测量的联合预算
    extra_phase_error: float = 0.0    # 已论证的额外 theta 误差上界（每次测量）
    max_alias_candidates: int = 10000
    seed: int = 20261012


def normalize_input(record: Mapping) -> dict:
    """既接收固定输入，也接收上一版 omega 程序的原始结果记录。"""
    if 'design' in record:
        d = record['design']
        get = (lambda k:d[k]) if isinstance(d,Mapping) else (lambda k:getattr(d,k))
        values = {k:float(get(k)) for k in
                  ('B_hat','C_hat','epsilon_B','epsilon_C','omega_min','omega_max')}
        values.update(omega_hat=float(record['omega_hat']),
                      frequency_error_bound=float(record['frequency_error_bound']))
    else:
        values = {k:float(record[k]) for k in
                  ('B_hat','C_hat','epsilon_B','epsilon_C','omega_hat',
                   'frequency_error_bound','omega_min','omega_max')}
    if not all(np.isfinite(v) for v in values.values()):
        raise ValueError('所有输入必须为有限数值。')
    if min(values['epsilon_B'],values['epsilon_C'],values['frequency_error_bound'])<0:
        raise ValueError('误差界必须非负。')
    wh, ew = values['omega_hat'], values['frequency_error_bound']
    wlo = max(values['omega_min'],wh-ew)
    whi = min(values['omega_max'],wh+ew)
    if wh<=0 or wlo<=0 or whi<wlo:
        raise ValueError('频率先验与估计区间不相容，或含非正频率。')
    values.update(omega_lower=wlo,omega_upper=whi,
                  omega_uncertainty=max(abs(wlo-wh),abs(whi-wh)),
                  C_abs_upper=abs(values['C_hat'])+values['epsilon_C'])
    return values


def systematic_phase_bound(kind: str, t: float, p: Mapping) -> float:
    """返回 theta（半 Ramsey 相位）的模型误差上界，非真误差。"""
    c, ew = p['C_abs_upper'],p['omega_uncertainty']
    if kind=='B':
        return c/p['omega_lower']*min(ew*t,1.)
    if kind=='C':
        coefficient_bias = ((np.pi/2-1)*c*ew/p['omega_hat']
                            + np.pi*c/12*(ew*t)**2)
        return 2*t/np.pi*coefficient_bias
    raise ValueError('kind 必须为 B 或 C。')


def synchronized_schedule(kind: str, p: Mapping, cfg: Config) -> list[int]:
    period = 2*np.pi/p['omega_hat']
    cap = cfg.max_B_time if kind=='B' else cfg.max_C_time
    max_m = int(np.floor(cap/period+1e-12))
    if kind=='C':
        # beta_C(t) 单调增加；整数周期二分选择，不使用真实频率。
        left,right=0,max_m
        while left<right:
            middle=(left+right+1)//2
            if systematic_phase_bound('C',middle*period,p)<=cfg.max_C_model_phase_error:
                left=middle
            else:
                right=middle-1
        max_m=left
    if max_m<1:
        return []
    periods=[]
    m=1
    while m<=max_m:
        periods.append(m)
        m*=2
    if periods[-1]!=max_m:
        periods.append(max_m)
    return periods


def square_wave_integrals(omega_hat: float, omega: float, m: int) -> tuple[float,float]:
    """真实模拟信号积分；脉冲时间只由已学习 omega_hat 决定。"""
    t=2*np.pi*m/omega_hat
    switches=(2*np.arange(2*m)+1)*np.pi/(2*omega_hat)
    edges=np.r_[0.,switches,t]
    signs=(-1.)**np.arange(len(edges)-1)
    g=float(np.dot(signs,np.diff(edges)))
    f=float(np.dot(signs,np.diff(np.sin(omega*edges)))/omega)
    return g,f


def simulate_ramsey_experiment(label: str, kind: str, m: int,
                               omega_hat: float, shots: Optional[int],
                               rng: np.random.Generator) -> dict:
    """可替换为实验接口。只返回测量期望/计数，不向估计器提供真实相位。"""
    term=TERM_BY_LABEL[label]
    t=2*np.pi*m/omega_hat
    if kind=='B':
        theta=term.B*t+term.C/term.omega*np.sin(term.omega*t)
    else:
        g,f=square_wave_integrals(omega_hat,term.omega,m)
        theta=term.B*g+term.C*f
    exact=np.array([np.cos(2*theta),np.sin(2*theta)])
    if shots is None:
        measured=exact
        counts=None
    else:
        counts=rng.binomial(shots,np.clip((1+exact)/2,0,1))
        measured=2*counts/shots-1
    return dict(R=float(measured[0]),S=float(measured[1]),counts_plus=counts)


def phase_from_measurement(measurement: Mapping, shots: Optional[int],
                           per_pair_failure: float) -> tuple[float,float]:
    phase=.5*math.atan2(measurement['S'],measurement['R'])
    if shots is None:
        return phase,1e-12
    # 两个 +/-1 均值的联合 Hoeffding 界，单次均值误差 <= eta。
    eta=math.sqrt(2*math.log(4/per_pair_failure)/shots)
    radius=math.sqrt(2)*eta
    phase_error=np.pi/2 if radius>=1 else .5*math.asin(radius)
    return phase,float(phase_error)


def merge_intervals(intervals: Sequence[tuple[float,float]]) -> list[tuple[float,float]]:
    result=[]
    for lo,hi in sorted(intervals):
        if result and lo<=result[-1][1]+1e-14:
            result[-1]=(result[-1][0],max(result[-1][1],hi))
        else:
            result.append((float(lo),float(hi)))
    return result


def intersect_phase_candidates(prior: Sequence[tuple[float,float]],phase: float,
                                gain: float,error: float,max_aliases: int):
    """theta = gain*coefficient + bounded error (mod pi).
    枚举所有与先验相交的分支；绝不选取最接近真实系数的分支。
    """
    intervals=[]
    count=0
    for lo,hi in prior:
        first=math.ceil((gain*lo-error-phase)/np.pi-1e-12)
        last=math.floor((gain*hi+error-phase)/np.pi+1e-12)
        count+=max(0,last-first+1)
        if count>max_aliases:
            raise RuntimeError('相位别名过多；需要更紧先验或更短起始时间。')
        for k in range(first,last+1):
            left=max(lo,(phase+k*np.pi-error)/gain)
            right=min(hi,(phase+k*np.pi+error)/gain)
            if left<=right:
                intervals.append((left,right))
    return merge_intervals(intervals)


def interval_estimate(intervals):
    # 保留所有候选，半径覆盖整个候选集合；绝不隐藏未消除的别名。
    lower,upper=intervals[0][0],intervals[-1][1]
    estimate=.5*(lower+upper)
    return estimate,.5*(upper-lower)


def refine_one_coefficient(label: str, kind: str, p: Mapping, cfg: Config,
                            periods: Sequence[int], per_pair_failure: float,
                            rng: np.random.Generator) -> dict:
    center,halfwidth=p[kind+'_hat'],p['epsilon_'+kind]
    intervals=[(center-halfwidth,center+halfwidth)]
    goal=cfg.target_B_error if kind=='B' else cfg.target_C_error
    history=[]
    status='target_reached' if halfwidth<=goal else 'time_or_frequency_limit'
    for m in periods:
        if interval_estimate(intervals)[1]<=goal:
            status='target_reached'
            break
        t=2*np.pi*m/p['omega_hat']
        measured=simulate_ramsey_experiment(label,kind,m,p['omega_hat'],cfg.shots,rng)
        phase,measurement_error=phase_from_measurement(measured,cfg.shots,per_pair_failure)
        model_error=systematic_phase_bound(kind,t,p)
        gain=t if kind=='B' else 2*t/np.pi
        updated=intersect_phase_candidates(intervals,phase,gain,
            model_error+measurement_error+cfg.extra_phase_error,cfg.max_alias_candidates)
        if not updated:
            raise RuntimeError(f'{label}/{kind}: 第 m={m} 周期的区间交集为空。'
                               '检查上游区间、噪声、控制误差或模型，不能强行输出高精度结果。')
        intervals=updated
        estimate,radius=interval_estimate(intervals)
        history.append(dict(m=m,t=t,theta_mod=phase,measurement=measured,
                            measurement_phase_error=measurement_error,
                            model_phase_error_bound=model_error,
                            estimate=estimate,error_bound=radius,
                            intervals=intervals.copy()))
    estimate,radius=interval_estimate(intervals)
    if radius<=goal:
        status='target_reached'
    elif not periods:
        status='no_admissible_period'
    elif len(intervals)>1:
        status='unresolved_phase_aliases'
    return dict(estimate=estimate,error_bound=radius,intervals=intervals,
                status=status,history=history,target_error=goal,
                max_interrogation_time=max((x['t'] for x in history),default=0.),
                sum_interrogation_times=sum(x['t'] for x in history),
                # R、S 必须使用独立制备；无限 shots 的总感知时间不定义为有限值。
                total_sensing_time=None if cfg.shots is None else
                    2*cfg.shots*sum(x['t'] for x in history))


def _refine_records(frequency_results: Optional[Mapping]=None,
                       config: Optional[Config]=None) -> dict:
    cfg=Config() if config is None else config
    if cfg.shots is not None and (not isinstance(cfg.shots,(int,np.integer)) or cfg.shots<=0):
        raise ValueError('shots 必须为正整数或 None。')
    if not 0<cfg.measurement_failure_probability<1:
        raise ValueError('测量失败概率必须在 (0,1)。')
    if min(cfg.target_B_error,cfg.target_C_error,cfg.max_B_time,cfg.max_C_time,
           cfg.max_C_model_phase_error)<=0 or cfg.extra_phase_error<0:
        raise ValueError('目标误差/时间/模型预算必须为正，额外相位误差非负。')
    inputs=DEFAULT_INPUTS if frequency_results is None else frequency_results
    params={lab:normalize_input(rec) for lab,rec in inputs.items()}
    schedules={(lab,kind):synchronized_schedule(kind,p,cfg)
               for lab,p in params.items() for kind in ('B','C')}
    # 预先为最多可能执行的全部测量分配失败概率，允许提前停止。
    max_pairs=sum(len(x) for x in schedules.values())
    per_pair_failure=cfg.measurement_failure_probability/max(1,max_pairs)
    results={}
    for index,(label,p) in enumerate(params.items()):
        rng=np.random.default_rng(cfg.seed+1000*index)
        b=refine_one_coefficient(label,'B',p,cfg,schedules[label,'B'],per_pair_failure,rng)
        c=refine_one_coefficient(label,'C',p,cfg,schedules[label,'C'],per_pair_failure,rng)
        result=dict(target_label=label,inputs=p,B=b,C=c,
                    B_hat_ref=b['estimate'],C_hat_ref=c['estimate'],
                    epsilon_B_ref=b['error_bound'],epsilon_C_ref=c['error_bound'])
        # 事后仿真诊断，不能用于控制流程或挑选结果。
        if label in TERM_BY_LABEL:
            truth=TERM_BY_LABEL[label]
            result['diagnostics']=dict(
                B_true=truth.B,C_true=truth.C,omega_true=truth.omega,
                B_error_before=abs(p['B_hat']-truth.B),
                C_error_before=abs(p['C_hat']-truth.C),
                B_error_after=abs(b['estimate']-truth.B),
                C_error_after=abs(c['estimate']-truth.C),
                initial_intervals_valid=(abs(p['B_hat']-truth.B)<=p['epsilon_B'] and
                    abs(p['C_hat']-truth.C)<=p['epsilon_C'] and
                    p['omega_lower']<=truth.omega<=p['omega_upper']),
                final_intervals_cover_truth=(any(lo<=truth.B<=hi for lo,hi in b['intervals'])
                    and any(lo<=truth.C<=hi for lo,hi in c['intervals'])))
        results[label]=result
    return results


def refine_single_qubit(frequency_result=None, config=None) -> dict:
    """直接接收单比特 omega 学习的结果字典；不传时使用上一步固定快照。"""
    record = DEFAULT_INPUT if frequency_result is None else frequency_result
    return _refine_records({'Z':record},config=config)['Z']


def print_single_qubit_result(result: Mapping, show_history: bool=False):
    p=result['inputs']
    print('单比特 B、C 精化输入')
    print(f'omega_hat            = {p["omega_hat"]:.12f}')
    print(f'frequency_error_bound= {p["frequency_error_bound"]:.12f}')
    print(f'omega_interval       = [{p["omega_lower"]:.12f}, {p["omega_upper"]:.12f}]')
    print('\n精化结果：误差界为当前候选区间的最大半径')
    print('Parameter    Initial estimate    Refined estimate      Initial bound      Refined bound')
    for kind in ('B','C'):
        r=result[kind]
        print(f'{kind:9s} {p[kind+"_hat"]:19.12f} {r["estimate"]:19.12f} '
              f'{p["epsilon_"+kind]:18.6e} {r["error_bound"]:18.6e}')
        print(f'  Candidate intervals: {r["intervals"]}')
    print('\n最长单次询问时间及目标精度检查')
    for kind in ('B','C'):
        r=result[kind]
        print(f'{kind}: T_max={r["max_interrogation_time"]:.6f}, '
              f'target={r["target_error"]:.3e}, '
              f'bound <= target: {r["error_bound"]<=r["target_error"]}')
        if show_history:
            for h in r['history']:
                print(f'  m={h["m"]:4d}, t={h["t"]:.6f}, '
                      f'estimate={h["estimate"]:.12f}, radius={h["error_bound"]:.3e}, '
                      f'branches={len(h["intervals"])}')
    d=result.get('diagnostics')
    if d:
        print('\n仿真真实误差（只用于事后比较，不是估计器输入）')
        for kind in ('B','C'):
            print(f'{kind}: true={d[kind+"_true"]:.12f}, '
                  f'before={d[kind+"_error_before"]:.6e}, '
                  f'after={d[kind+"_error_after"]:.6e}')
        print('Initial intervals valid   =',d['initial_intervals_valid'])
        print('Final intervals valid     =',d['final_intervals_cover_truth'])


if __name__=='__main__':
    config=Config(
        target_B_error=1e-4,
        target_C_error=1e-4,
        max_B_time=1024.,
        max_C_time=64.,
        shots=None,  # 改为正整数（例如 200000）可模拟有限次测量
    )
    result=refine_single_qubit(config=config)
    print_single_qubit_result(result,show_history=False)
